import threading
import time

start = time.perf_counter()

def do_something():
    print("Sleeping 1 seconds ... ")
    time.sleep(1)
    print("Done Sleeping ...")

do_something()
finish = time.perf_counter()
print(f"Finished in {finish-start} second(s)")

start = time.perf_counter()
t1 = threading.Thread(target=do_something)
t2 = threading.Thread(target=do_something)
t1.start()
t2.start()


t1.join()
t2.join()


finish = time.perf_counter()
print(f"Finished in {finish-start} second(s)")

