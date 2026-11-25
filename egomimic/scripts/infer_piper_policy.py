"""PiPER live inference or recorded-input offline evaluation with one pipeline."""


def main(argv=None):
    # Spawn imports this entry module in each recording/device child. Keep
    # model, Torch and CUDA-related imports in the actual parent invocation.
    from egomimic.robot.piper_pico.live_policy import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
