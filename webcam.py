import numpy as np
import face_recognition as fr
import cv2
from engine import get_Rostos

def reconhecer_pessoas():
    rostos_conhecidos, nome_dos_rostos = get_Rostos()
    reconhecido = False

    video_capture = cv2.VideoCapture(0)
    processar_este_frame = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Erro ao capturar imagem da webcam!")
            break


        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if processar_este_frame:
            localizacao_dos_rostos = fr.face_locations(rgb_frame)
            rostos_desconhecidos = fr.face_encodings(rgb_frame, localizacao_dos_rostos)

            for (top, right, bottom, left), rosto_desconhecido in zip(localizacao_dos_rostos, rostos_desconhecidos):
                resultado = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)
                face_distance = fr.face_distance(rostos_conhecidos, rosto_desconhecido)
                melhor_id = np.argmin(face_distance)

                if resultado[melhor_id]:
                    nome = nome_dos_rostos[melhor_id]
                    reconhecido = True
                else:
                    nome = "Desconhecido"


                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Desenha o quadrado e o nome
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                if reconhecido:

                    video_capture.release()
                    cv2.destroyAllWindows()
                    return True


        processar_este_frame = not processar_este_frame


        cv2.imshow("Webcam_faceRecognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()

    return False


if __name__ == "__main__":
    resultado = reconhecer_pessoas()
    print("Rosto reconhecido?", resultado)
