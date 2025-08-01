import json
import threading
import cv2
from ultralytics import YOLO
import socket
import os


def loadConfig():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	config_path = os.path.join(script_dir, 'config.json')
	with open(config_path, 'r') as configFile:
		configData = json.load(configFile)

	print("Configuración: ")
	print(configData)
	return configData


def draw_label(frame, label, position=(10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Coordenadas del rectángulo blanco
    x, y = position
    w, h = text_size
    cv2.rectangle(frame, (x - 5, y - h - 5), (x + w + 5, y + 5), (255, 255, 255), -1)
    cv2.putText(frame, label, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)



def verifyObjectLocation(pt, object_centers, frame_index):
	global frame_rects

	rect_color = (0, 0, 255)
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
			rect_color = (0, 255, 0)
			break
	
	frame_rects[frame_index].append((pt[0], pt[1], rect_color))
	return is_all_test_passed


def drawList(elements, frame):
	pts1 = [500, 50]
	pts2 = [630, 80]
	for element in elements:
		color = (0, 255, 0) if element['state'] else (255, 255, 255)
		cv2.rectangle(frame, pts1, pts2, color, -1)
		cv2.putText(frame, element['classname'], (pts1[0] + 5, pts1[1]+20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
		pts1[1] += 50
		pts2[1] += 50


def getIdData(id):
	global configData
	
	roiData = configData['ROI']
	if id in roiData["center_cam"]:
		# print(f'ID {id}: center_cam')
		return (0, roiData["center_cam"][id])
	
	if id in roiData["right_cam"]:
		# print(f'ID {id}: right_cam')
		return (1, roiData["right_cam"][id])
	
	if id in roiData["left_cam"]:
		# print(f'ID {id}: left_cam')
		return (2, roiData["left_cam"][id])
	
	print(f'ID {id}: not found in any ROI config')
	return (-1, [])


def getPredictions():
	global frame1, frame2, frame3 

	centers1 = []
	centers2 = []
	centers3 = []
	# Predict using yolov8
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

	return results_list


def detectForPackingModel(model, ids):
	global configData, frame_rects

	frame_rects = [[], [], []]
	response = f'El modelo {model} no existe en el archivo de configuracion '
	if model in configData['models']:
		predictions = getPredictions()
		model_data = configData['models'][model]
		response = f'{model},'
		for id in ids:
			frame_index, idData = getIdData(id)
			# print(id in model_data)
			# print(frame_index >= 0)
			if (id in model_data) and frame_index >= 0:
				if verifyObjectLocation(idData, predictions[frame_index]['centers'], frame_index):
					response += '1,'
				else:
					response += '0,'
			else:
				response += '2,'
		
	return response[:-1]


# PROD: xxx.xxx.xxx.xxx # DEBUG: Set this for prod
# TEST: 192.168.1.94
def start_server(host, port=4455):
	EOF = "EOF"
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server_socket.bind((host, port))
	server_socket.listen(1)
	print(f'Servidor escuchando en {host}:{port}')

	while True:
		data = None
		client_socket, client_address = server_socket.accept()
		print(f'Conexión aceptada de {client_address}')
		while True:
			data = client_socket.recv(1024).decode('utf-8')
			if data:
				print(f'Mensaje recibido: {data}')
				received_string = data.strip() 
				splitted_data = received_string.split(",")
				model, *ids = splitted_data
				if model and len(ids) > 0:
					print(f'Model: {model}')
					print(f'IDs: {ids}')
					response = detectForPackingModel(model, ids)
					client_socket.sendall(response.encode('utf-8'))
			else:
				print('No se recibió ningún dato.')

			client_socket.close()
			print(f'Conexión cerrada con {client_address}')
			break


# Load Yolov8 model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolov8_custom.pt")
model = YOLO(model_path).to("cpu")

# DEBUG comment this block
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap3 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
	print("No se puede acceder a la cámara.")
	exit()

if __name__ == '__main__':
	global frame1, frame2, frame3, configData, frame_rects
	frame_rects = [[], [], []] # Array elements must be (pt1, pt2, color)
	configData = loadConfig()
	server_thread = threading.Thread(target= lambda: start_server(host=configData['connection']['ip']))
	server_thread.daemon = True
	server_thread.start()

	while True:
		# Capturar un cuadro de la cámara
		ret1, frame1_temp = cap.read()
		ret2, frame2_temp = cap2.read()
		ret3, frame3_temp = cap3.read()

		# Si no se ha capturado correctamente, continuar con el siguiente ciclo
		if not ret1 or not ret2 or not ret3:
			print("Error al capturar la imagen.")
			break

		# Use images as frames for development
		# frame1_temp = cv2.imread('./img0.png')
		# frame2_temp = cv2.imread('./img0_env.png')
		# frame3_temp = cv2.imread('./img0_prop.png')

		unordered_frames = [frame1_temp, frame2_temp, frame3_temp]

		frame1 = unordered_frames[configData["cameras"]["center"]]
		frame2 = unordered_frames[configData["cameras"]["right"]]
		frame3 = unordered_frames[configData["cameras"]["left"]]

		frames = [frame1, frame2, frame3]
		for index, frame_data in enumerate(frame_rects):
			for rect in frame_data:
				cv2.rectangle(frames[index], rect[0], rect[1], rect[2], 2)

		draw_label(frame1, "center")
		draw_label(frame2, "right")
		draw_label(frame3, "left")

		cv2.imshow('frame', frame1)
		cv2.imshow('frame2', frame2)
		cv2.imshow('frame3', frame3)

		# Salir del bucle si se presiona la tecla 'q'
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Liberar la captura y cerrar las ventanas
	cap.release()
	cap2.release()
	cap3.release()
	cv2.destroyAllWindows()
