import cv2

def get_chain_code(image_path):
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(original_image, 70, 255, 0)

    # Find the starting point
    start_point = None
    for i, row in enumerate(img):
        for j, value in enumerate(row):
            if value == 255:
                start_point = (i, j)
                break
        if start_point:
            break

    directions = [0, 1, 2, 7, 3, 6, 5, 4]
    dir2idx = dict(zip(directions, range(len(directions))))

    change_j = [-1, 0, 1, -1, 1, -1, 0, 1]
    change_i = [-1, -1, -1, 0, 0, 1, 1, 1]

    border = []
    chain = []
    curr_point = start_point

    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
        if 0 <= new_point[0] < img.shape[0] and 0 <= new_point[1] < img.shape[1] and img[new_point] != 0:
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break  # Fixed indentation error

    count = 0
    while curr_point != start_point:
        # Search direction starts
        b_direction = (direction + 5) % 8
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)

        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
            if 0 <= new_point[0] < img.shape[0] and 0 <= new_point[1] < img.shape[1] and img[new_point] != 0:
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break

        if count == 1000:
            break
        count += 1

    return chain
