"""Closed-loop bean-scoop client with the LED cue driven by the CLIENT (2026-09-05, user: "a new client called
led client: every time I can set x for blink x times, and when it starts running, after getting the first position
and the policy starts running, auto blink").

Same robot/camera/RTC plumbing and server contract as examples/yam/client_memory_v5.py (imported), plus the cue
protocol of the data collection (gello launch_yaml_memory_led.py, same defaults: pre-wait 0.5-1.5 s, green
0.3 s on with 0.4-0.8 s gaps, GO delay 0.6-2.0 s, then yellow stays on):

    connect -> ramp to the first policy target -> memory reset -> control loop starts
           -> the cue thread starts on the FIRST control step: pre-wait, x green blinks, GO delay, yellow ON
    keys:  1 / 2 / 3 = x for the NEXT episode      r = reset memory, LEDs off, new episode (cue restarts)
           q = quit (LEDs off, arms left in place)

The overlay adds a LED line (x, cue phase, green/yellow state); the recording gets a sidecar
<record>.led.json with every cue event (wall time + control step) per episode, so the sentence stream can be
checked against the true blink times afterwards.

Server (GPU box):  JOB=<job> bash cluster_v5/serve_v5_job.sh <ckpt dir> pi05_yam_mem_v5_beansB6 8000
Client (robot computer; needs gello_software with gello.utils.led_cue, openpi_client, opencv):
    python examples/yam/client_memory_v5_led.py --host 10.79.12.149 --port 8000 --blinks 2
    python examples/yam/client_memory_v5_led.py --host 10.79.12.149 --port 8000 --blinks 2 --no-led   (cue printed only)
    python examples/yam/client_memory_v5_led.py --host 10.79.12.149 --port 8000 --dry-run              (no hardware)
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import pathlib
import random
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import client_memory_v5 as base  # noqa: E402

from openpi_client import action_chunk_broker  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402

# The beans prompt, defined here too so an older client_memory_v5.py on the robot computer (bin task only) still works.
BEANS_PROMPT = getattr(base, "BEANS_PROMPT", "scoop the beans into the tray as many times as the green light blinked")


@dataclasses.dataclass
class Args(base.Args):
    host: str = "10.79.12.149"
    """iris-hgx-2 (the 2xH200 job 17284681); the bin-task default was iris-hgx-1."""
    prompt: str = BEANS_PROMPT
    steps_between_inference: int = 5
    """The beans models train at memory stride 5 (one memory tick every 5 controls = 1/6 s)."""

    # --- LED cue ---
    blinks: int = 2
    """x for the first episode (1-3): the green LED blinks x times, the robot should scoop x times.
    Keys 1/2/3 in the window change x for the next episode (applied at the next 'r')."""
    led_port: str | None = None
    """Serial port of the LED board; unset = auto-detect the CH340 by USB vendor id (stable by-id path)."""
    led_baudrate: int = 9600
    no_led: bool = False
    """Run the cue timing without hardware (prints ON/OFF instead of writing to the board)."""
    pre_wait_min: float = 0.5
    pre_wait_max: float = 1.5
    """Delay (s) between the loop start / reset and the first blink."""
    blink_on_s: float = 0.3
    """How long the green LED stays on per blink."""
    gap_min: float = 0.4
    gap_max: float = 0.8
    """Gap (s) between blinks (randomized, as in the collection)."""
    go_delay_min: float = 0.6
    go_delay_max: float = 2.0
    """Delay (s) between the last blink and the yellow GO signal."""
    timing_seed: int | None = None
    """Seed for the cue timing RNG; unset = fresh timings."""


# --- LED board -------------------------------------------------------------------------------------------------
CH340_VENDOR_ID = "1a86"  # QinHeng CH340, the tower light's USB-serial chip


class _PrintLed:
    """Stand-in for gello.utils.led_cue.FakeLedCue when gello is not importable (prints only)."""

    def green_on(self):
        print("[LED] green ON")

    def green_off(self):
        print("[LED] green off")

    def yellow_on(self):
        print("[LED] yellow ON")

    def all_off(self):
        print("[LED] all off")

    def close(self):
        pass


def _serial_ports() -> list[tuple[str, str, str]]:
    out = []
    for dev in sorted(pathlib.Path("/dev").glob("ttyUSB*")) + sorted(pathlib.Path("/dev").glob("ttyACM*")):
        try:
            props = subprocess.run(
                ["udevadm", "info", "-q", "property", "-n", str(dev)], capture_output=True, text=True, timeout=5
            ).stdout
        except Exception:
            continue
        vid, by_id = "", ""
        for line in props.splitlines():
            if line.startswith("ID_VENDOR_ID="):
                vid = line.split("=", 1)[1]
            elif line.startswith("DEVLINKS="):
                for link in line.split("=", 1)[1].split():
                    if link.startswith("/dev/serial/by-id/"):
                        by_id = link
        out.append((str(dev), vid, by_id or str(dev)))
    return out


def _autodetect_led_port() -> str:
    ports = _serial_ports()
    hits = [by_id for _dev, vid, by_id in ports if vid == CH340_VENDOR_ID]
    if len(hits) == 1:
        logging.info("Auto-detected LED board (CH340) at %s", hits[0])
        return hits[0]
    listing = ", ".join(f"{dev}(vendor {vid or '?'})" for dev, vid, _ in ports) or "none"
    if not hits:
        raise ValueError(f"No CH340 tower light found. Ports present: {listing}. Pass --led-port or --no-led.")
    raise ValueError(f"Multiple CH340 devices found ({', '.join(hits)}); pass --led-port.")


def _setup_led(args: Args):
    if args.no_led:
        try:
            from gello.utils.led_cue import FakeLedCue

            return FakeLedCue()
        except Exception:
            return _PrintLed()
    from gello.utils.led_cue import LedCue

    port = args.led_port or _autodetect_led_port()
    led = LedCue(port, baudrate=args.led_baudrate)
    logging.info("LED ready on %s", port)
    return led


# --- cue thread ------------------------------------------------------------------------------------------------
class LedCue:
    """Runs one episode's cue on its own thread; the control loop keeps stepping throughout."""

    def __init__(self, led, args: Args, step_of):
        self._led = led
        self._args = args
        self._step_of = step_of  # callable -> current control step (for the event log)
        self._rng = random.Random(args.timing_seed)
        self._thread: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self.lock = threading.Lock()
        self.x = 0
        self.phase = "idle"
        self.green = False
        self.yellow = False
        self.record: dict = {}
        self.episodes: list[dict] = []

    def _log(self, name: str) -> None:
        now = time.time()
        with self.lock:
            self.record.setdefault("events", []).append(
                {"name": name, "t": now, "t_rel": now - self.record["t0"], "step": int(self._step_of())}
            )

    def _set(self, phase: str | None = None, green: bool | None = None, yellow: bool | None = None) -> None:
        with self.lock:
            if phase is not None:
                self.phase = phase
            if green is not None:
                self.green = green
            if yellow is not None:
                self.yellow = yellow

    def _all_off(self) -> None:
        try:
            self._led.all_off()
        except Exception as e:  # noqa: BLE001
            logging.warning("LED all_off failed: %s", e)
        self._set(green=False, yellow=False)

    def _run(self, x: int, stop: threading.Event) -> None:
        a = self._args
        self._all_off()
        self._log("init")
        pre = self._rng.uniform(a.pre_wait_min, a.pre_wait_max)
        self.record["pre_wait_s"] = pre
        self._set(phase="pre-wait")
        if stop.wait(pre):
            return self._abort()
        gaps = []
        for i in range(x):
            self._led.green_on()
            self._set(phase=f"blink {i + 1}/{x}", green=True)
            self._log("blink_on")
            aborted = stop.wait(a.blink_on_s)
            self._led.green_off()
            self._set(green=False)
            self._log("blink_off")
            if aborted:
                return self._abort()
            gap = self._rng.uniform(a.gap_min, a.gap_max)
            gaps.append(gap)
            if stop.wait(gap):
                return self._abort()
        self.record["gaps_s"] = gaps
        go_delay = self._rng.uniform(a.go_delay_min, a.go_delay_max)
        self.record["go_delay_s"] = go_delay
        self._set(phase="go-delay")
        logging.info("%d blink(s) done; GO in %.2f s", x, go_delay)
        if stop.wait(go_delay):
            return self._abort()
        self._led.yellow_on()
        self._set(phase="GO", yellow=True)
        self._log("go")
        self.record["completed"] = True
        logging.info(">>> GO <<<  (x=%d) the policy should scoop %d time(s) and put the scoop down", x, x)

    def _abort(self) -> None:
        self._all_off()
        self._set(phase="aborted")

    def start(self, x: int) -> None:
        self.cancel()
        with self.lock:
            self.x = x
            self.record = {"x": x, "t0": time.time(), "step0": int(self._step_of()), "completed": False, "events": []}
            self.phase = "starting"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(x, self._stop), name="led-cue", daemon=True)
        self._thread.start()
        logging.info("LED cue started: x=%d (pre-wait %.1f-%.1f s, blink %.2f s, gaps %.1f-%.1f s, GO delay %.1f-%.1f s)",
                     x, self._args.pre_wait_min, self._args.pre_wait_max, self._args.blink_on_s,
                     self._args.gap_min, self._args.gap_max, self._args.go_delay_min, self._args.go_delay_max)

    def cancel(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        self._stop = None
        with self.lock:
            if self.record:
                self.record["t_end"] = time.time()
                self.record["step_end"] = int(self._step_of())
                self.episodes.append(dict(self.record))
                self.record = {}
        self._all_off()
        self._set(phase="idle")

    def overlay_line(self, next_x: int) -> str:
        with self.lock:
            g = "ON" if self.green else "off"
            y = "ON" if self.yellow else "off"
            return f"LED x={self.x} | {self.phase} | green {g} yellow {y} | next x={next_x} (keys 1/2/3)"

    def close(self) -> None:
        self.cancel()
        try:
            self._led.close()
        except Exception:  # noqa: BLE001
            pass


class _Display(base._Display):  # noqa: SLF001
    """Base window plus keys 1/2/3 (x for the next episode)."""

    def __init__(self, window: str = "pi05 yam memory v5 - beans LED closed loop"):
        self.next_x_requested: int | None = None
        super().__init__(window)

    def _loop(self) -> None:
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        while not self._stop.is_set():
            with self._lock:
                img = None if self._img is None else self._img.copy()
            if img is not None:
                cv2.imshow(self._window, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("r"):
                self.reset_requested.set()
            elif key == ord("q"):
                self.quit_requested.set()
            elif key in (ord("1"), ord("2"), ord("3")):
                self.next_x_requested = int(chr(key))
        cv2.destroyAllWindows()


def validate_v5_metadata(metadata: dict, args: Args) -> None:
    """The base client's contract checks, with the beans prompt and stride 5 accepted (an older
    client_memory_v5.py on the robot computer knows only the bin-task prompts / stride 15)."""
    if not metadata.get("memory_v5_sentence_bank"):
        raise ValueError(f"this client needs a v5 sentence-bank server; server config {metadata.get('config_name')!r}")
    if metadata.get("memory_architecture") != "v32_layer8_dual_query":
        raise ValueError(f"unexpected memory_architecture {metadata.get('memory_architecture')!r}")
    if metadata.get("action_horizon") != args.action_horizon:
        raise ValueError(f"server action_horizon is {metadata.get('action_horizon')!r}, client {args.action_horizon}")
    if metadata.get("rtc_enabled") is not True:
        raise ValueError("the server checkpoint is not RTC-trained (rtc_enabled must be true)")
    if metadata.get("rtc_delay_semantics") != "inclusive_max":
        raise ValueError(f"unexpected RTC delay semantics {metadata.get('rtc_delay_semantics')!r}")
    trained_max_delay = metadata.get("rtc_max_delay")
    if not isinstance(trained_max_delay, int) or args.max_async_delay_steps > trained_max_delay:
        raise ValueError(f"client max_async_delay_steps={args.max_async_delay_steps} exceeds the server's RTC maximum ({trained_max_delay!r})")
    training_stride = metadata.get("memory_stride_frames")
    if training_stride != args.steps_between_inference:
        raise ValueError(f"server memory_stride_frames is {training_stride!r}, but the client replans every {args.steps_between_inference} steps")
    if args.prompt != BEANS_PROMPT:
        raise ValueError(f"prompt {args.prompt!r} is not the beans training prompt {BEANS_PROMPT!r}")


def main(args: Args) -> None:
    if not 1 <= args.blinks <= 3:
        raise ValueError("--blinks must be 1, 2 or 3 (the training range)")
    try:
        ws_client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port, ping_timeout=args.ping_timeout)
    except TypeError:
        logging.warning("openpi_client without ping_timeout support; make sure the server was warmed up")
        ws_client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = ws_client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    validate_v5_metadata(metadata, args)
    policy = action_chunk_broker.RealtimeActionChunkBroker(
        ws_client,
        action_horizon=args.action_horizon,
        steps_between_inference=args.steps_between_inference,
        initial_delay_steps=args.initial_delay_steps,
        delay_tolerance_steps=args.delay_tolerance_steps,
        max_async_delay_steps=args.max_async_delay_steps,
        delay_buffer_size=args.delay_buffer_size,
    )
    if args.dry_run:
        try:
            base._run_dry(ws_client, policy, args)  # noqa: SLF001
        finally:
            policy.close()
        return

    led_board = _setup_led(args)  # open the board BEFORE the arms so a wrong port fails early

    from gello.cameras.realsense_camera import RealSenseCamera
    from gello.env import RobotEnv
    from gello.robots.robot import BimanualRobot
    from gello.robots.yam import YAMRobot

    robot = BimanualRobot(YAMRobot(channel=args.can_left), YAMRobot(channel=args.can_right))
    assert robot.num_dofs() == base.BIMANUAL_DOF, f"expected 14 DOF, got {robot.num_dofs()}"
    camera_dict = {
        "top_camera": RealSenseCamera(device_id=args.top_camera_serial),
        "left_camera": RealSenseCamera(device_id=args.left_camera_serial),
        "right_camera": RealSenseCamera(device_id=args.right_camera_serial),
    }
    env = RobotEnv(robot, control_rate_hz=args.hz, camera_dict=camera_dict)
    obs = env.get_obs()
    for key in ("top_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
        assert key in obs, f"missing camera obs '{key}'"
    assert np.asarray(obs["joint_positions"]).shape == (base.BIMANUAL_DOF,)

    # --- Ramp to the first inferred target (LEDs off: the model must see "no blink yet") ---
    ws_client.infer({"reset_memory": True})
    logging.info("Ramping to first policy target (LEDs off)...")
    first_target = np.asarray(policy.infer(base._obs_to_request(obs, args.prompt))["actions"], dtype=np.float64)  # noqa: SLF001
    for _ in range(25):
        obs = env.get_obs()
        cur = np.asarray(obs["joint_positions"], dtype=np.float64)
        if float(np.abs(first_target - cur).max()) < 1e-2:
            break
        env.step(base._clamp_joint_delta(first_target, cur, args.max_joint_delta))  # noqa: SLF001
    policy.reset()
    ws_client.infer({"reset_memory": True})
    logging.info("Memory reset -- episode starts fresh (bank blank).")

    display = _Display() if args.show else None
    step_holder = [0]
    cue = LedCue(led_board, args, lambda: step_holder[0])
    next_x = args.blinks

    writer = None
    record_path = ""
    frames_written = 0
    if args.record:
        record_path = args.record_path
        if not record_path:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
            os.makedirs(args.record_dir, exist_ok=True)
            record_path = os.path.join(args.record_dir, f"memory_v5_led_closedloop_{stamp}.mp4")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(record_path)) or ".", exist_ok=True)
        first_frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
        writer = base._H264Writer(record_path, first_frame.shape[1], first_frame.shape[0], args.hz)  # noqa: SLF001
        logging.info("Recording top camera (H.264) to %s @ %.1f Hz", record_path, args.hz)

    logging.info("Starting RTC control loop (replan every %d steps, horizon %d), prompt %r; cue x=%d starts now",
                 args.steps_between_inference, args.action_horizon, args.prompt, next_x)
    obs = env.get_obs()
    subtask, seen, held = "", "", ""
    cue.start(next_x)  # the policy is running: the blinks start after the pre-wait
    try:
        for step in range(args.max_steps):
            step_holder[0] = step
            if display is not None and display.quit_requested.is_set():
                logging.info("Quit requested from the display window.")
                break
            if display is not None and display.next_x_requested is not None:
                next_x = display.next_x_requested
                display.next_x_requested = None
                logging.info("Next episode: x=%d", next_x)
            if display is not None and display.reset_requested.is_set():
                display.reset_requested.clear()
                cue.cancel()
                policy.reset()
                ws_client.infer({"reset_memory": True})
                logging.info("Memory reset (keyboard) -- new episode, x=%d.", next_x)
                cue.start(next_x)

            result = policy.infer(base._obs_to_request(obs, args.prompt))  # noqa: SLF001
            action = np.asarray(result["actions"], dtype=np.float64)
            subtask = str(result.get("subtask", subtask))
            if "bank" in result:
                seen, held = base.memory_readout(result)
            status = f"step {step} | ticks {result.get('writes', 0)} | r=reset q=quit"
            frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
            img = base._overlay(frame, subtask, [seen, held, cue.overlay_line(next_x), status])  # noqa: SLF001
            if display is not None:
                display.update(img)
            if writer is not None:
                writer.write(img)
                frames_written += 1
            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = base._clamp_joint_delta(action, cur, args.max_joint_delta)  # noqa: SLF001
            obs = env.step(action)
            if step % args.steps_between_inference == 0:
                logging.info("  step %d | %s | subtask: %s | %s | %s", step, cue.overlay_line(next_x), subtask, seen, held)
    except KeyboardInterrupt:
        logging.info("Interrupted by user -- stopping (arms left in place).")
    finally:
        cue.close()
        if writer is not None:
            writer.release()
            logging.info("Saved recording: %s (%d frames)", record_path, frames_written)
            side = record_path + ".led.json"
            with open(side, "w") as f:
                json.dump({"episodes": cue.episodes, "args": dataclasses.asdict(args)}, f, indent=2, default=str)
            logging.info("Saved LED cue log: %s", side)
        if display is not None:
            display.close()
        policy.close()
        logging.info("Control loop finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
