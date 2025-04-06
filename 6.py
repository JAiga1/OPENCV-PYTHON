import cv2
import easyocr

# Load the image
image = cv2.imread('v4.jpg')

# Convert to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding based on background color
# For white text on black background
_, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)

# For black text on white background (invert thresholding)
# _, binary_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY_INV)

# Apply morphological closing to enhance the letters (optional)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
processed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Save the processed image temporarily
cv2.imwrite('processed_v.jpg', processed_image)

# Perform OCR on the processed image
reader = easyocr.Reader(['en'])
result = reader.readtext('processed_v.jpg')

# Print the OCR results
print(result)

# Optionally display the processed image
# cv2.imshow('Processed Image', processed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
