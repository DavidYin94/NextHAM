import timeit
import functools
import logging
import os

# Setup logger for timing
timing_logger = logging.getLogger("timing_logger")
timing_logger.setLevel(logging.INFO)

# Create a file handler
log_file = os.path.join(os.getcwd(), "timing_report.log")
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
timing_logger.addHandler(file_handler)

def time_execution(func):
    """Decorator to measure the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        execution_time = end_time - start_time
        timing_logger.info(f"Function '{func.__name__}' in '{func.__module__}' executed in {execution_time:.6f} seconds")
        print(f"[Timing] Function '{func.__name__}' executed in {execution_time:.6f} seconds", flush=True)
        return result
    return wrapper
