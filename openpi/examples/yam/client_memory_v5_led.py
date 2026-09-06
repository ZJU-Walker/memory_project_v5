"""STANDALONE closed-loop bean-scoop client with the LED cue driven by the client (2026-09-05).

No import from client_memory_v5.py: everything it needs is in this file. Dependencies on the robot computer:
`openpi_client`, `gello_software` (YAMRobot, RealSenseCamera, RobotEnv, gello.utils.led_cue), opencv, numpy, tyro,
and ffmpeg on PATH for the recording.

Protocol (same numbers as the data collection, gello launch_yaml_memory_led.py):
    connect -> ramp to the first policy target (LEDs off) -> memory reset -> control loop starts
           -> the cue thread starts on the FIRST control step: pre-wait 0.5-1.5 s, x green blinks (0.3 s on,
              0.4-0.8 s gaps), GO delay 0.6-2.0 s, yellow ON (stays on until reset/quit)
    keys:  1 / 2 / 3 = x for the NEXT episode      r = reset memory, LEDs off, new episode (cue restarts)
           q = quit (LEDs off, arms left in place)

The overlay shows the decoded sentence + confidence (* = committed this tick), the bank contents, and a LED line
(x, cue phase, green/yellow state). The recording gets a sidecar <video>.led.json with every cue event
(wall time + control step) per episode.

Server (GPU box):  JOB=<job> bash cluster_v5/serve_v5_job.sh <ckpt dir> pi05_yam_mem_v5_beansB6 8000
Client:
    python client_memory_v5_led.py --host 10.79.12.149 --port 8000 --blinks 2
    python client_memory_v5_led.py --host 10.79.12.149 --port 8000 --blinks 2 --no-led   (cue printed only)
    python client_memory_v5_led.py --host 10.79.12.149 --port 8000 --dry-run              (no hardware)
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import pathlib
import random
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

BEANS_PROMPT = "scoop the beans into the tray as many times as the green light blinked"
BIMANUAL_DOF = 14  # 6 arm joints + 1 gripper per arm, concat(left, right)
CH340_VENDOR_ID = "1a86"  # QinHeng CH340, the tower light's USB-serial chip


@dataclasses.dataclass
class Args:
    # --- Policy server (remote GPU box) ---
    host: str = "10.79.12.149"
    """iris-hgx-2 (the 2xH200 job 17284681)."""
    port: int = 8000
    ping_timeout: float = 600.0
    """Websocket keepalive timeout (s); generous in case the server was started without --warmup."""

    # --- Inference / control ---
    action_horizon: int = 50
    """Full model action horizon. Must match the server checkpoint."""
    steps_between_inference: int = 5
    """The beans models train at memory stride 5: replan (= one memory tick) every 5 controls @ 30 Hz."""
    initial_delay_steps: int = 6
    max_async_delay_steps: int = 6
    delay_tolerance_steps: int = 0
    delay_buffer_size: int = 8
    max_steps: int = 12000
    hz: float = 30.0
    prompt: str = BEANS_PROMPT
    max_joint_delta: float = 1.0
    """Per-step safety clamp: cap |target - current| across all joints to this many radians."""

    # --- LED cue ---
    blinks: int = 2
    """x for the first episode (1-3). Keys 1/2/3 in the window set x for the next episode (applied at 'r')."""
    led_port: str | None = None
    """Serial port of the LED board; unset = auto-detect the CH340 by USB vendor id (stable by-id path)."""
    led_baudrate: int = 9600
    no_led: bool = False
    """Run the cue timing without the board (prints ON/OFF)."""
    pre_wait_min: float = 0.5
    pre_wait_max: float = 1.5
    blink_on_s: float = 0.3
    gap_min: float = 0.4
    gap_max: float = 0.8
    go_delay_min: float = 0.6
    go_delay_max: float = 2.0
    timing_seed: int | None = None

    # --- Display / recording ---
    show: bool = True
    record: bool = True
    record_dir: str = "eval"
    record_path: str = ""

    # --- Hardware (defaults from gello configs/yam_left.yaml) ---
    can_left: str = "can_left"
    can_right: str = "can_right"
    top_camera_serial: str = "409122273280"
    left_camera_serial: str = "409122271088"
    right_camera_serial: str = "409122271086"

    # --- Debug ---
    dry_run: bool = False
    """Skip hardware and LEDs: validate the RTC replan and obs/action/subtask/memory contract."""
    dry_run_steps: int = 40


# --- recording / overlay / window ------------------------------------------------------------------------------
class _H264Writer:
    """Encode RGB frames to an H.264 mp4 via the system ffmpeg."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH -- needed to encode the recording")
        self._proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", f"{fps}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", path],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        self._proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()


def memory_readout(result: dict) -> tuple[str, str]:
    """Two overlay lines: the sentence decoded now (confidence, `*` = committed this tick) and the bank contents."""
    confidence = result.get("subtask_confidence")
    memory = result.get("memory") or {}
    mark = "*" if memory.get("committed") else ""
    conf = f"({float(confidence):.2f})" if confidence is not None else ""
    line_seen = f"sees: {result.get('subtask', '')!s}{conf}{mark}  (* = committed now)"
    bank = result.get("bank") or []
    shown = " | ".join(str(b) for b in bank) if bank else "-"
    line_held = f"bank[{len(bank)}]: {shown}  commits {result.get('writes', 0)}"
    return line_seen, line_held


def _overlay(frame_rgb: np.ndarray, subtask: str, lines: list[str]) -> np.ndarray:
    img = np.ascontiguousarray(frame_rgb).copy()
    cv2.putText(img, f"subtask: {subtask}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 2, cv2.LINE_AA)
    for k, line in enumerate(lines):
        cv2.putText(img, line, (12, 66 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 60), 1, cv2.LINE_AA)
    return img


class _Display:
    """Window thread showing the latest overlaid frame; captures r (reset), q (quit), 1/2/3 (next x)."""

    def __init__(self, window: str = "pi05 yam memory v5 - beans LED closed loop"):
        self._window = window
        self._lock = threading.Lock()
        self._img: np.ndarray | None = None
        self.reset_requested = threading.Event()
        self.quit_requested = threading.Event()
        self.next_x_requested: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, img_rgb: np.ndarray) -> None:
        with self._lock:
            self._img = img_rgb

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

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


# --- observation / safety / contract ---------------------------------------------------------------------------
def _obs_to_request(obs: dict, prompt: str) -> dict:
    return {
        "observation/state": np.asarray(obs["joint_positions"], dtype=np.float32),
        "observation/image": image_tools.convert_to_uint8(obs["top_camera_rgb"]),
        "observation/left_wrist_image": image_tools.convert_to_uint8(obs["left_camera_rgb"]),
        "observation/right_wrist_image": image_tools.convert_to_uint8(obs["right_camera_rgb"]),
        "prompt": prompt,
    }


def _clamp_joint_delta(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    delta = target - current
    m = float(np.abs(delta).max())
    if m > max_delta:
        delta = delta / m * max_delta
    return current + delta


def validate_v5_metadata(metadata: dict, args: Args) -> None:
    """Fail before touching hardware when the client/server contracts differ."""
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
        raise ValueError(
            f"client max_async_delay_steps={args.max_async_delay_steps} exceeds the server's RTC maximum ({trained_max_delay!r})"
        )
    training_stride = metadata.get("memory_stride_frames")
    if training_stride != args.steps_between_inference:
        raise ValueError(
            f"server memory_stride_frames is {training_stride!r}, but the client replans every {args.steps_between_inference} steps"
        )
    if args.prompt != BEANS_PROMPT:
        raise ValueError(f"prompt {args.prompt!r} is not the beans training prompt {BEANS_PROMPT!r}")


def _run_dry(ws_client, policy, args: Args) -> None:
    if args.dry_run_steps <= args.steps_between_inference + args.max_async_delay_steps:
        raise ValueError("dry_run_steps must exceed steps_between_inference + max_async_delay_steps")
    rng = np.random.default_rng(0)
    logging.info("Dry run: memory reset + %d random control observations...", args.dry_run_steps)
    logging.info("  reset: %s", ws_client.infer({"reset_memory": True}))
    for i in range(args.dry_run_steps):
        example = {
            "observation/state": rng.random(BIMANUAL_DOF).astype(np.float32),
            "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "prompt": args.prompt,
        }
        result = policy.infer(example)
        action = np.asarray(result["actions"])
        assert action.shape == (BIMANUAL_DOF,), f"expected (14,) per broker step, got {action.shape}"
        assert np.all(np.isfinite(action)), "non-finite action returned"
        assert isinstance(result.get("subtask"), str), f"missing subtask, got {result.get('subtask')!r}"
        for key in ("subtask_confidence", "bank", "memory", "writes"):
            assert key in result, f"v5 server response lacks {key!r}: {sorted(result)}"
        seen, held = memory_readout(result)
        logging.info("  step %d: subtask=%r | %s | %s", i, result["subtask"], seen, held)
    logging.info("Dry run OK -- RTC replan and obs/action/subtask/memory contract match.")


# --- LED board -------------------------------------------------------------------------------------------------
class _PrintLed:
    """Stand-in when gello's FakeLedCue is not importable (prints only)."""

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
        except Exception:  # noqa: BLE001
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
    """Find the CH340 tower light by USB vendor id (ttyUSB numbers move between reboots)."""
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
        except Exception:  # noqa: BLE001
            return _PrintLed()
    from gello.utils.led_cue import LedCue

    port = args.led_port or _autodetect_led_port()
    led = LedCue(port, baudrate=args.led_baudrate)
    logging.info("LED ready on %s", port)
    return led


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
        a = self._args
        logging.info("LED cue started: x=%d (pre-wait %.1f-%.1f s, blink %.2f s, gaps %.1f-%.1f s, GO delay %.1f-%.1f s)",
                     x, a.pre_wait_min, a.pre_wait_max, a.blink_on_s, a.gap_min, a.gap_max, a.go_delay_min, a.go_delay_max)

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


# --- main ------------------------------------------------------------------------------------------------------
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
            _run_dry(ws_client, policy, args)
        finally:
            policy.close()
        return

    led_board = _setup_led(args)  # open the board BEFORE the arms so a wrong port fails early

    from gello.cameras.realsense_camera import RealSenseCamera
    from gello.env import RobotEnv
    from gello.robots.robot import BimanualRobot
    from gello.robots.yam import YAMRobot

    robot = BimanualRobot(YAMRobot(channel=args.can_left), YAMRobot(channel=args.can_right))
    assert robot.num_dofs() == BIMANUAL_DOF, f"expected 14 DOF, got {robot.num_dofs()}"
    camera_dict = {
        "top_camera": RealSenseCamera(device_id=args.top_camera_serial),
        "left_camera": RealSenseCamera(device_id=args.left_camera_serial),
        "right_camera": RealSenseCamera(device_id=args.right_camera_serial),
    }
    env = RobotEnv(robot, control_rate_hz=args.hz, camera_dict=camera_dict)
    obs = env.get_obs()
    for key in ("top_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
        assert key in obs, f"missing camera obs '{key}'"
    assert np.asarray(obs["joint_positions"]).shape == (BIMANUAL_DOF,)

    # --- Ramp to the first inferred target (LEDs off: the model must see "no blink yet") ---
    ws_client.infer({"reset_memory": True})
    logging.info("Ramping to first policy target (LEDs off)...")
    first_target = np.asarray(policy.infer(_obs_to_request(obs, args.prompt))["actions"], dtype=np.float64)
    for _ in range(25):
        obs = env.get_obs()
        cur = np.asarray(obs["joint_positions"], dtype=np.float64)
        if float(np.abs(first_target - cur).max()) < 1e-2:
            break
        env.step(_clamp_joint_delta(first_target, cur, args.max_joint_delta))
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
        writer = _H264Writer(record_path, first_frame.shape[1], first_frame.shape[0], args.hz)
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

            result = policy.infer(_obs_to_request(obs, args.prompt))
            action = np.asarray(result["actions"], dtype=np.float64)
            subtask = str(result.get("subtask", subtask))
            if "bank" in result:
                seen, held = memory_readout(result)
            status = f"step {step} | ticks {result.get('writes', 0)} | r=reset q=quit"
            frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
            img = _overlay(frame, subtask, [seen, held, cue.overlay_line(next_x), status])
            if display is not None:
                display.update(img)
            if writer is not None:
                writer.write(img)
                frames_written += 1
            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = _clamp_joint_delta(action, cur, args.max_joint_delta)
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
