for i in range(0,18000):
    
    if i >= 3600 and i < 5400:
        img = np.rot90(org_train_x[i].reshape(28,28),k =3, axes=(0,1))
        img = np.fliplr(img)
    elif i >= 9000 and i < 10800:
        img = np.rot90(org_train_x[i].reshape(28,28),k =3, axes=(0,1))
        img = np.fliplr(img)
    elif i >= 12600 and i < 14400:
        img = np.rot90(org_train_x[i].reshape(28,28),k =3, axes=(0,1))
        img = np.fliplr(img)
    elif i>= 14400 and i < 16200:
        img = np.rot90(org_train_x[i].reshape(28,28),k =3, axes=(0,1))
        img = np.fliplr(img)
    else:
        img = org_train_x[i].reshape(28,28)
    height, width = img.shape[:2]
    dst = cv2.resize(img, (6*width, 6*height), interpolation = cv2.INTER_CUBIC)

    x = np.pad(dst,pad_width=40, mode='constant', constant_values=[0])
    im = Image.fromarray(x)
    im.save("sample.png")
    plt.imshow(x)
    plt.show()
    