import torch
import cv2
from ultralytics import YOLO
import time

# Cargar el modelo YOLOv8
model = YOLO("model_navico_trained.pt").to("cpu")

# debug: uncomment for prod
# Iniciar la captura de video (cámara por defecto)
# cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
# cap2 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap3 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# debug: uncomment for prod
# Verificar si la cámara está abierta correctamente
# if not cap.isOpened():
#     print("No se puede acceder a la cámara.")
#     exit()

pts1 = [
			((250, 110), (500, 440), 'box_xl'),
		]

pts2 = [
			((200, 100), (360, 200), 'medium_box'),
			((180, 50), (370, 120), 'envelope_xl'),
			((300, 280), (340, 390), 'envelope_m'),
			# Envelope S
		]

pts3 = [
			((50, 60), (140, 130), 'small_box'), 
			((150, 290), (360, 400), 'propeller')
		]


##################### DEBUG FUNCTIONS: NOT FOR PROD #####################
def switchPassPts2(state):
	global pts2
	if state:
		pts2 = [
			((200, 100), (360, 200), 'medium_box'),
			((180, 50), (370, 120), 'envelope_xl'),
		]
	else:
		pts2 = [
			((200, 100), (360, 200), 'medium_box'),
			((180, 50), (370, 120), 'envelope_xl'),
			((300, 280), (340, 390), 'envelope_m'),
		]

# switchPassPts2(False)

def switchPassPts3(state):
	global pts3
	if state:
		pts3 = [
			((50, 60), (140, 130), 'propeller'),
		]
	else:
		pts3 = [ 
			((150, 290), (360, 400), 'propeller')
		]

# switchPassPts3(True)

statePts2 = False
statePts3 = False
##################### END OF DEBUG FUNCTIONS: NOT FOR PROD #####################

def verifyObjectLocation(positions, object_centers, frame):
	elements = []
	count = 0

	for pt in positions:
		rectColor = (0, 0, 255)
		classname = pt[2]
		is_all_test_passed = False
		for center in object_centers:
			x, y = center['center']
			is_correct_class = center['classname'] == classname
			if pt[2] in ['envelope_m', 'envelope_s']:
				is_correct_class = center['classname'] in ['envelope_m', 'envelope_s']
			is_in_range_x = pt[0][0] < x < pt[1][0]
			is_in_range_y = pt[0][1] < y < pt[1][1]
			is_all_test_passed = is_in_range_x and is_in_range_y and is_correct_class
			if is_all_test_passed:
				rectColor = (0, 255, 0)
				count += 1
				break
		
		elements.append({'classname': classname, 'state':is_all_test_passed})
		cv2.rectangle(frame, pt[0], pt[1], rectColor, 2)
	print(elements)
	drawList(elements, frame)
	return count == len(positions)


def drawList(elements, frame):
	pts1 = [500, 50]
	pts2 = [630, 80]
	for element in elements:
		color = (0, 255, 0) if element['state'] else (255, 255, 255)
		cv2.rectangle(frame, pts1, pts2, color, -1)
		cv2.putText(frame, element['classname'], (pts1[0] + 5, pts1[1]+20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
		pts1[1] += 50
		pts2[1] += 50


def drawListColor(pts, frame, color):
	pts1 = [500, 50]
	pts2 = [630, 80]
	for pt in pts:
		cv2.rectangle(frame, pt[0], pt[1], color, 2)
		cv2.rectangle(frame, pts1, pts2, color, -1)
		cv2.putText(frame, pt[2], (pts1[0] + 5, pts1[1]+20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
		pts1[1] += 50
		pts2[1] += 50
	


is_first_two_passed = False
is_all_passed = False

passed_pause_start = 0

while True:

	# debug: comment for prod
	switchPassPts2(statePts2)
	switchPassPts3(statePts3)

	centers1 = []
	centers2 = []
	centers3 = []

	# debug: uncomment for prod
	# Capturar un cuadro de la cámara
	# ret, frame1 = cap.read()
	# ret, frame2 = cap2.read()
	# ret, frame3 = cap3.read()
	
	# debug: uncomment for prod
	# # Si no se ha capturado correctamente, continuar con el siguiente ciclo
	# if not ret:
	#     print("Error al capturar la imagen.")
	#     break
	
	# debug: use images as input
	frame1 = cv2.imread('img0.png')
	frame2 = cv2.imread('img0_env.png')
	frame3 = cv2.imread('img0_prop.png')

	# Realizar la predicción con YOLOv8
	results1 = model.predict(frame1, conf=0.5)
	results2 = model.predict(frame2, conf=0.1)
	results3 = model.predict(frame3, conf=0.5)

	results_list = [
					{'results': results1, 'frame': frame1, 'centers': centers1}, 
					{'results': results2, 'frame': frame2, 'centers': centers2}, 
					{'results': results3, 'frame': frame3, 'centers': centers3}
				]
	for results in results_list:
		for result in results['results']:
			for det in result.boxes:
				classname = result.names[int(det.cls)]
				x1, y1, x2, y2 = det.xyxy.int().tolist()[0]
				center_point = (x1 + (x2-x1)//2, y1 + (y2-y1)//2)
				results['centers'].append({'center':center_point, 'classname':classname})
				cv2.circle(results['frame'], center_point, 5, (0, 255, 0), -1)

	

	# Dibujar las cajas delimitadoras en la imagen
	#frame_with_boxes1 = results1[0].plot()  # La función plot dibuja las cajas de detección sobre la imagen
	#frame_with_boxes2 = results2[0].plot()
	#frame_with_boxes3 = results3[0].plot() 

	if not is_first_two_passed:
		state_1 = verifyObjectLocation(pts1, centers1, frame1)
		state2 = verifyObjectLocation(pts2, centers2, frame2)
		is_first_two_passed = state_1 and state2
	elif not is_all_passed:
		drawListColor(pts1, frame1, (0, 204, 255))
		drawListColor(pts2, frame2, (0, 204, 255))
		state3 = verifyObjectLocation(pts3, centers3, frame3)
		if state3:
			is_all_passed = True
			passed_pause_start = time.time()

	if is_all_passed:
		if time.time() - passed_pause_start < 5:
			drawListColor(pts1, frame1, (0, 255, 0))
			drawListColor(pts2, frame2, (0, 255, 0))
			drawListColor(pts3, frame3, (0, 255, 0))
		else:
			is_all_passed = False
			is_first_two_passed = False


	cv2.imshow('frame', frame1)
	cv2.imshow('frame2', frame2)
	cv2.imshow('frame3', frame3)

	key = cv2.waitKey(1)

	if key == ord('2'):
		statePts2 = not statePts2

	if key == ord('3'):
		statePts3 = not statePts3

	# Salir del bucle si se presiona la tecla 'q'
	if key & 0xFF == ord('q'):
		break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
