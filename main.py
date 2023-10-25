import cv2
def similarity_orb(img1, img2):
    orb = cv2.ORB.create()

    # detect keypoints and descriptors
    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    # brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # find the similarity
    matches = bf.match(desc_a, desc_b)

    # compute the similarity
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

# open the images
image1 = cv2.imread("55.jpg", 1)
image2 = cv2.imread("50.jpg", 1)

similar_orb = similarity_orb(image1, image2)
print(similar_orb)


# if the similarity between two images are big
if similar_orb >= 0.5:
    print("this is an id card")

else:
    print("this is not an id card")

