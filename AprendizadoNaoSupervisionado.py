import cv2
import numpy as np
import matplotlib.pyplot as plt

def carregar_imagem(caminho): #carregar imagem no arquivo
    return cv2.imread(caminho)

def exibir_imagem(imagem, titulo='Imagem'): #exibir uma imagem usando matplotlib
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.show()

def aplicar_kmeans(imagem, k): #aplicar o k-médias a imagem
    altura, largura, _ = imagem.shape
    dados = imagem.reshape((altura * largura, 3))

    # Aplicar k-médias
    kmeans = cv2.kmeans(np.float32(dados), k, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    # Atualizar cores na imagem
    centros = np.uint8(kmeans[2])
    resultado = centros[kmeans[1].flatten()]
    resultado = resultado.reshape((altura, largura, 3))

    return resultado

def calcular_propriedades(imagem): #calcular as propriedades das imagens
    altura, largura, _ = imagem.shape
    total_pixels = altura * largura

    # Calcular média e desvio padrão das intensidades de cor
    media_cor = np.mean(imagem, axis=(0, 1))
    desvio_padrao_cor = np.std(imagem, axis=(0, 1))

    # Encontrar cores únicas na imagem
    cores_unicas = np.unique(imagem.reshape(-1, imagem.shape[2]), axis=0)
    quantidade_cores = len(cores_unicas)

    return {
        'altura': altura,
        'largura': largura,
        'total_pixels': total_pixels,
        'media_cor': media_cor,
        'desvio_padrao_cor': desvio_padrao_cor,
        'quantidade_cores': quantidade_cores
    }

def main():
    # Carregar imagem
    caminho_imagem = 'C:\\Users\\gui_f\\OneDrive\\IA\\raposa.png'  #alterar a imagem a ser carregada
    imagem_original = carregar_imagem(caminho_imagem)

    # Exibir imagem original
    exibir_imagem(imagem_original, 'Imagem Original')

    # Calcular propriedades da imagem original
    propriedades_original = calcular_propriedades(imagem_original)
    print("Propriedades da Imagem Original:")
    print(propriedades_original)

    # Aplicar k-médias com diferentes valores de k
    valores_k = [2, 4, 8, 16, 32, 64, 128]
    for k in valores_k:
        imagem_kmeans = aplicar_kmeans(imagem_original, k)

        # Exibir imagem após k-médias
        exibir_imagem(imagem_kmeans, f'Imagem após k-médias (k={k})')

        # Calcular propriedades da imagem após k-médias
        propriedades_kmeans = calcular_propriedades(imagem_kmeans)
        print(f"Propriedades da Imagem após k-médias (k={k}):")
        print(propriedades_kmeans)

if __name__ == "__main__":
    main()