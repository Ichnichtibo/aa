import threading
import time

done = False

def worker(text):
    counter = 0
    while True:
        time.sleep(1)
        counter+=1
        print(f"{text}: {counter}")



a = threading.Thread(target=worker,daemon=True,args=("ABC",))
a.start()
print(a.getName())


input("Press enter to quit")
done = True

