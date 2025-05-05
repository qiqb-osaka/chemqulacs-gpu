class TimeoutException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "Terminating optimization: time limit reached"
