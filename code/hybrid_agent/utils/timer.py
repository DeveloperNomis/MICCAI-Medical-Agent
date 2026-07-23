import time

class Timer:

    def __init__(self):
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        if name not in self.start_times:
            return None

        duration = time.perf_counter() - self.start_times[name]
        print(f"[TIMER] {name}: {duration:.3f} seconds")
        return duration