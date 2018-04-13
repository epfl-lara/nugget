
def timeout(func, duration):
    """Timeout.

    Adapted from StackOverflow:
    https://stackoverflow.com/questions/492519/timeout-on-a-function-call
    """

    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(duration)
    try:
        result = func()
    except TimeoutError as exc:
        result = None
    finally:
        signal.alarm(0)

    return result