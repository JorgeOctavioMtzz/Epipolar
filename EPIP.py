import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_keypoints_and_matches(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return pts1, pts2

def compute_fundamental_matrix(pts1, pts2):
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
    return F

def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    return E

def extract_translation_and_rotation(E, K, pts1, pts2):
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def main():
    img1 = cv2.imread('Epipolar\img1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('Epipolar\img2.jpg', cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error al cargar las imágenes.")
        exit()

    fx = 600  # Distancia focal en píxeles a lo largo del eje x
    fy = 600  # Distancia focal en píxeles a lo largo del eje y
    cx = 320  # Coordenada x del punto principal
    cy = 240  # Coordenada y del punto principal

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    pts1, pts2 = extract_keypoints_and_matches(img1, img2)
    F = compute_fundamental_matrix(pts1, pts2)
    E = compute_essential_matrix(F, K)
    R, t = extract_translation_and_rotation(E, K, pts1, pts2)

    print('Matriz de rotación:', R)
    print('Matriz de traslación:', t)

    # Plotting
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB))
    ax[0].scatter(pts1[:, 0, 0], pts1[:, 0, 1], c='r', s=10)
    ax[0].set_title('Imagen 1')
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB))
    ax[1].scatter(pts2[:, 0, 0], pts2[:, 0, 1], c='b', s=10)
    ax[1].set_title('Imagen 2')
    plt.show()

if __name__ == '__main__':
    main()
