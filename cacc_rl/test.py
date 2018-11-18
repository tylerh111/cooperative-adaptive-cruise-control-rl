

import threading
import time


def calc_square(lst):
	print('starting to square')
	for i in lst:
		time.sleep(0.2)
		print(i,':',i ** 2)

def calc_cube(lst):
	print('starting to cube')
	for i in lst:
		time.sleep(0.2)
		print(i,':',i ** 3)

arr = [0,1,2,3,4,5,6,7,8,9]


t1 = threading.Thread(target=calc_square, args=(arr,))
t2 = threading.Thread(target=calc_cube, args=(arr,))

print('starting computation')

t1.start()
t2.start()

#print('joining')

t1.join()
t2.join()

print('joined')




