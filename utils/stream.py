def stream_augment(img, pixelate=1, blur=1):
    print(f"Applying stream artifacts...")

    augments = [
        lambda img: blur_image(img, strength=blur),
        lambda img: pixelate_image(img, strength=pixelate),
    ]
    random.shuffle(augments)

    for augment in augments:
        augment(img)

def pixelate_image(img, strength=1):
    """
        Pixelate an image by a random factor in-place.
        Width and height are pixalated separately.
    Args:
        image (ndarray): Input image to be pixelated in-place
        strength (float, optional): 0 means no change, 1 means max pixelation

    Returns:
        [type]: [description]
    """

    height, width = img.shape[:2]

    # 0 => 1 (no change)
    # 1 => .4 (pixelate by 40%)
    fr = remap(strength, 0, 1, 1, .4)

    fr_w = random.uniform(fr, 1)
    fr_h = random.uniform(fr, 1)

    # print("Pixelate")
    # print(fr_w, fr_h)

    # desired "pixelated" size
    w,h = (width*fr_w, height*fr_h)

    # Resize input to "pixelated" size
    temp = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

    # Blow it back up over input image
    cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST, dst=img)

def blur_image(img, strength=1):
    """
    Blur an image by a random factor in-place.
    Widght and height are blurred separatey.
    Args:
        image (ndarray): Input image to be blurred in-place.
        strength (int, optional): 0 means no change, 1 means max blur. Defaults to 1.
    """

    max_kernel_size = 7

    # 1 => 7
    # 0 => 1
    fr = remap(strength, 0, 1, 1, max_kernel_size)
    fr_w = random.randrange(1, fr)
    fr_h = random.randrange(1, fr)

    # print("Blur")
    # print(fr_w, fr_h)

    cv2.boxFilter(img, ddepth=-1, ksize=(fr_w,fr_h), dst=img)