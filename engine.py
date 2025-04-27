import face_recognition as fr

def reconheceRosto(url_foto):
    foto = fr.load_image_file(url_foto)
    rostos = fr.face_encodings(foto)

    if len(rostos) > 0:
        return True, rostos[0]
    return False, None

def get_Rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []

    sucesso, rosto_Pessoa = reconheceRosto("img.jpg") #Adicione uma imagem para o app ter uma base de reconhecimento
    if sucesso and rosto_Pessoa is not None:
        rostos_conhecidos.append(rosto_Pessoa)
        nomes_dos_rostos.append("Pessoa")
    else:
        print("Não foi possível reconhecer o rosto na imagem")

    return rostos_conhecidos, nomes_dos_rostos
