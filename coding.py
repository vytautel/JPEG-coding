# %%

from numpy import ndarray, dstack, uint8, float32, empty, full, array, round, floor, clip, maximum, tile, mean, zeros, zeros_like
'''
dstack - stacks arrays in the third dimensions
empty - creates uninitialized array
full - creates array filled with specific value
'''
from skimage.metrics import mean_squared_error # for MSE between two images
import matplotlib.pyplot as plt
from PIL import Image # Python Imaging Library (PIL), useful for image processing
import cv2, math 
import os
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
import heapq
from collections import defaultdict, Counter
import numpy as np

# display image from pixels
def show_image(pixels: ndarray):
    if pixels.ndim == 2: # for grayscale
        plt.imshow(pixels, cmap='gray')  
    elif pixels.ndim == 3 and pixels.shape[2] == 3: # for RGB
        plt.imshow(pixels) 
    else:
        raise ValueError("Unsupported image format")
    
    plt.axis('off')  # Hide the axis
    plt.show()
	
# Returns a generator that yields the R, G, and B channels of the image.
def components(pixels: ndarray):
    #print(pixels)
    #print(pixels.shape)
    return (pixels[:, :, i] for i in range(3))

# Clips values in the array to be within the range 0 to 255. 
# Useful for ensuring pixel values remain valid after transformations.
def clip_color(array):
	return clip(array, 0, 255)

# https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
# Converts an image from YCbCr (luminance-chrominance color space) to RGB color space using a standard formula.
def YCbCr_to_rgb(pixels: ndarray):
	(y, cb, cr) = components(pixels) # Extracts Y, Cb, and Cr channels.

	r = y + 1.402 * (cr - 128)
	g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
	b = y + 1.772 * (cb - 128)

	return clip_color(dstack((r, g, b))) 
		# clip_color ensures that the values are clipped to the 0-255 range.

# Converts an RGB image to YCbCr (luminance-chrominance color space) color space using a standard formula.
def rgb_to_YCbCr(pixels: ndarray):
	(r, g, b) = components(pixels) # Extracts R, G, and B channels.

	y = (0.299 * r) + (0.587 * g) + (0.114 * b)  # Y channel
	cb = 128 - (0.168736 * r) - (0.331264 * g) + (0.5 * b)  # Cb channel
	cr = 128 + (0.5 * r) - (0.418688 * g) - (0.081312 * b)  # Cr channel

	return  clip_color(dstack((y, cb, cr)))  
		# clip_color ensures pixel values are within the valid range 0-255

def visualize_channel(comp, i, grey_comp):
    """Visualizes the Y, Cb, or Cr channel, filling unused channels with gray."""
    if i == 0:
        return comp  # For the Y channel, return it as is
    else:
        # Create a tuple of channels, replacing unused channels with grey_comp
        channels = tuple(comp if i == j else grey_comp for j in range(3))
        # Convert the YCbCr channels back to RGB
        return YCbCr_to_rgb(dstack(channels))

# https://en.wikipedia.org/wiki/Chroma_subsampling
#  the practice of encoding images by implementing less resolution for chroma information than for luma information,
#  taking advantage of the human visual system's lower acuity for color differences than for luminance
# boxFilter() applies a box filter to subsample the Cb and Cr channels (reduces resolution). Only the Y channel is not subsampled.
# [::2, ::2]: Reduces the resolution by a factor of 2 in both dimensions.
#  Centers the data around 0 for further processing.
# Subsamples (reduces resolution) for the Cb and Cr channels using a box filter, leaving Y unchanged.

# sampling ratios: 444 (4:4:4) - no subsampling
# 422 (4:2:2) - subsamples the chrominance channels horizontally, halving their horizontal resolution, while maintaining full vertical resolution.
# 420 (4:2:0) - subsamples both horizontally and vertically, reducing the chrominance resolution in both dimensions.
def subsample_YCbCr(YCbCr_pixels, sampleMode = 420):
	# Initialize an empty list to store the subsampled channels
    subsampled = []
    
    # Iterate over each channel in YCbCr pixels
	# When i == 0, comp refers to the Y channel, i == 1: Cb, i == 2: Cr
    for i, comp in enumerate(components(YCbCr_pixels)):
        if i == 0 or sampleMode == 444:
            subsampled.append(comp) # Keep channel unchanged
            
        elif sampleMode == 422:
            filtered_channel = cv2.boxFilter(comp, ddepth=-1, ksize=(2, 1))  # Filter in the horizontal direction
            reduced_channel = filtered_channel[:, ::2]  # Downsample horizontally
            subsampled.append(reduced_channel)
            
        elif sampleMode == 420:
            # Apply box filter and reduce resolution for Cb and Cr channels
			# when ddepth=-1, the output image will have the same depth as the source
			# Chrominance channels are smoothened by averaging the pixel values in a small 2×2 neighborhood
			# [::2, ::2] part takes every second element along both rows and columns (every second pixel in both dimensions),
			# which halves the resolution in both directions.
            filtered_channel = cv2.boxFilter(comp, ddepth=-1, ksize=(2, 2))
            reduced_channel = filtered_channel[::2, ::2]
            subsampled.append(reduced_channel)

    # Center the data around 0 by subtracting 128 from all channels
    return [channel - 128 for channel in subsampled]

# https://en.wikipedia.org/wiki/Quantization_(image_processing)#Quantization_matrices
# used for compression, less shades for certain (for example red) color.

# These matrices (QY for luminance and QC for chroma) are used in compression
# to reduce the amount of data for less critical information.
# The QY matrix is used for the Y (luminance) channel, 
# and QC is used for the Cb and Cr (chrominance) channels.

# Quantization Formula: floor((S * q) + 0.5) scales and 
# rounds the values, ensuring they don't go below 1 (maximum(..., 1)).
# The number 0.5 in the quantization calculation is used as a rounding factor

# Function to calculate the quantization matrices based on the scaling factor S
def calculate_quantization_matrices(S, blockSize = 8):
    QY_table = array([
		[16, 11, 10, 16, 24,  40,  51,  61],
		[12, 12, 14, 19, 26,  48,  60,  55],
		[14, 13, 16, 24, 40,  57,  69,  56],
		[14, 17, 22, 29, 51,  87,  80,  62],
		[18, 22, 37, 56, 68,  109, 103, 77],
		[24, 35, 55, 64, 81,  104, 113, 92],
		[49, 64, 78, 87, 103, 121, 120, 101],
		[72, 92, 95, 98, 112, 100, 103, 99]
	])
    QC_table = array([
		[17, 18, 24, 47, 99, 99, 99, 99],
		[18, 21, 26, 66, 99, 99, 99, 99],
		[24, 26, 56, 99, 99, 99, 99, 99],
		[47, 66, 99, 99, 99, 99, 99, 99],
		[99, 99, 99, 99, 99, 99, 99, 99],
		[99, 99, 99, 99, 99, 99, 99, 99],
		[99, 99, 99, 99, 99, 99, 99, 99],
		[99, 99, 99, 99, 99, 99, 99, 99]
	])
    
    if (blockSize == 4):
	    # Crop the matrices
        QY_table = QY_table[:blockSize, :blockSize]
        QC_table = QC_table[:blockSize, :blockSize]

    QY = maximum(floor((S * QY_table) + 0.5), 1)
    QC = maximum(floor((S * QC_table) + 0.5), 1)
    return QY, QC

# https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga85aad4d668c01fbd64825f589e3696d4
# Constructs the coefficient matrix C used in the Discrete Cosine Transform (DCT).
def contruct_C(QY, B):
	C = empty(QY.shape, float32) 
	for j in range(B): # B - block size
		alpha = math.sqrt((1 + int(j != 0)) / B) # normalization factor α
		beta = (math.pi * j) / (B * 2) # frequency 𝛽
		for k in range(B):
			C[j, k] = alpha * math.cos(beta * (2 * k + 1))
	return C

zigzag_index = array([
	[ 0, 1, 5, 6,14,15,27,28],
	[ 2, 4, 7,13,16,26,29,42],
	[ 3, 8,12,17,25,30,41,43],
	[ 9,11,18,24,31,40,44,53],
	[10,19,23,32,39,45,52,54],
	[20,22,33,38,46,51,55,60],
	[21,34,37,47,50,56,59,61],
	[35,36,48,49,57,58,62,63]
])
# traverse block in zigzam 
def zigzag_scan(block, block_size = 8):
     
    assert block.shape == (block_size, block_size), "Block must be 8x8"
    zigzag_result = [block[row, col] for row, col in zip(*np.unravel_index(zigzag_index.flatten(), (block_size, block_size)))]
    return zigzag_result

# Reverses the zigzag ordering and places elements back in an 8x8 block.
def reverse_zigzag(input, block_size = 8):
    
 	# Ensure input is a numpy array
    input = np.array(input).flatten()
    zigzag_index_local = zigzag_index.flatten()

    # Check if input is smaller than block_size^2
    if input.shape[0] < block_size ** 2:
        input = np.pad(input, (0, block_size ** 2 - input.shape[0]), mode='constant')

    # Initialize an empty 8x8 block    
    block = np.zeros((block_size, block_size), dtype=input.dtype)

    # Fill the block based on the zigzag order
    for idx in range(block_size ** 2):
        try:
            row, col = divmod(zigzag_index_local[idx], block_size)
            block[row, col] = input[idx]
        except IndexError:
            print("Index error: make sure zigzag_index is properly defined.")

    return block
	
# Run-Length Encoding (RLE)
def run_length_encode(zigzag_list):
    encoded = []
    count = 0
    for value in zigzag_list:
        if value == 0:
            count += 1
        else:
            encoded.append((count, value))  # (zero run-length, non-zero value)
            count = 0
    # Append remaining zeros
    if count > 0:	
        encoded.append((count, 0))
    return encoded

# Build a Huffman tree
def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

# Encode symbols using the Huffman code
def huffman_encode(symbols, huffman_table):
     # Create a dictionary from the Huffman table for easy lookup
    huffman_dict = dict(huffman_table)
    
    # Encode only the second value (coefficient) of each (run_length, coefficient) pair
    encoded_string = ''.join([huffman_dict[symbol[1]] for symbol in symbols])
    
    return encoded_string

def huffman_decode(encoded_string, huffman_table):
    # Invert the huffman_table to create a decoding dictionary (bit sequence -> symbol)
    decoding_dict = {v: k for k, v in huffman_table}
    
    decoded_symbols = []
    current_bits = ""
    
    # Iterate through the encoded bit string
    for bit in encoded_string:
        current_bits += bit
        
        # If the current bit sequence is in the decoding dictionary, decode it
        if current_bits in decoding_dict:
            decoded_symbol = decoding_dict[current_bits]
            decoded_symbols.append(decoded_symbol)
            current_bits = ""  # Reset current bits after a match
    
    return decoded_symbols

   # Pads the block to the specified block size (e.g., 8x8) using zeros.
def pad_block(block, block_size):
    padded_block = zeros((block_size, block_size), dtype=block.dtype)
    padded_block[:block.shape[0], :block.shape[1]] = block
    return padded_block

# Full encoding for one channel (e.g., Y, Cb, Cr)
# YCbCr to RGB → DCT + Quantization → Inverse Zigzag Scan → RLE Encoding → Huffman Encoding

def compress_channel(quant_matrix, channel, block_size=8):
    height, width = channel.shape
    
    channel = encode_PRIMARY(quant_matrix, channel)
    
    compressed_data = []
    
    # Loop over the image in block_size x block_size blocks
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):            
            block = channel[i:i + block_size, j:j + block_size]
            
            # Apply zigzag scan to the DCT block
            zigzag_list = zigzag_scan(block)
            
            # Apply run-length encoding (RLE) to the zigzag-ordered coefficients
            #rle_list = run_length_encode(zigzag_list)

            compressed_data.extend(zigzag_list)
            # Store the RLE-compressed block data
           # compressed_data.extend(dct_block)
    
    # Get frequency counts for Huffman encoding
    #compressed_data = [item[1] for item in compressed_data]
    # Extract the second element (the actual values) for frequency calculation
    #values = [value for _, value in compressed_data]

	# Calculate frequencies of the values
    #frequencies = Counter(array(values))
    
	# Example values of compressed_data: [(0, 14.0), (0, -5.0), (7, -1.0), (2, 1.0), (3, 1.0), (0, -6.0), (1, 1.0), (0, -1.0), (11, 1.0), (0, -1.0), (30, 0)]
	# 1st value - number of consecutive zeros (or runs of similar values) before encountering a significant non-zero value.
    # 2nd value - This represents the non-zero coefficient encountered after the run of zeros, comes from the quantized DCT coefficients.
    
    #frequencies = Counter(compressed_data)
    
    # Build a Huffman tree based on the frequency counts
    #huffman_table = build_huffman_tree(frequencies)
    
    # Huffman encode the RLE-compressed data
    #huffman_encoded = huffman_encode(compressed_data, huffman_table)
    
   # print(huffman_encoded)
   # return huffman_encoded, huffman_table

    print(channel)
    return channel, []

# Discrete Cosine Transform (DCT)
# C is the DCT coefficient matrix, block - image block
def dct(block):
	return C @ block @ C.T

# inverse DCT
def idct(block):
	return C.T @ block @ C

# Defines the encoding process for a channel using DCT and quantization.
# q- quantization matrix, channel - 2D array (grayscale img)
def encode_PRIMARY(q, channel):
	trans = empty(channel.shape, float32) # transformed (compressed) values

	# looping trough B x B size blocks
	for row in range(0, channel.shape[0], B):
		for col in range(0, channel.shape[1], B):
			trans[row:row+B, col:col+B] = round(
				dct(channel[row:row+B, col:col+B]) / q)

	return trans

def encode_block(block, quant_matrix):    
    quantized_block = round(dct(block) / quant_matrix)
    return quantized_block

# Inside the decode function, for each block of the image (row:row+B, col:col+B), 
# it performs the inverse DCT (idct) to reconstruct the original image block. 
# It multiplies by the quantization matrix q, adds back 128 
# (since the YCbCr channels were centered around 128),
# and clips the values to the valid range of 0 to 255 using clip_color.
def decode_PRIMARY(q, channel):
	trans = empty(channel.shape, float32)
	for row in range(0, channel.shape[0], B):
		for col in range(0, channel.shape[1], B):
			trans[row:row+B, col:col+B] = clip_color(
				idct(channel[row:row+B, col:col+B] * q) + 128)

	# Resizes decoded channel back to original dimensions
	# Rounds pixels to integers.
	return round(cv2.resize(trans, dimensions[::-1]))

  #  Applies inverse DCT (IDCT) to each 8x8 block in the image channel.
def apply_idct(image_channel, block_size):
    height, width = image_channel.shape
    reconstructed_channel = zeros_like(image_channel, dtype=float)

    # Loop over each block of size block_size x block_size
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            # Extract the current block
            block = image_channel[i:i + block_size, j:j + block_size]
            
            # Apply the IDCT to the current block using your idct function
            idct_block = idct(block)
            
            # Store the reconstructed block in the corresponding position
            reconstructed_channel[i:i + block_size, j:j + block_size] = idct_block

    return reconstructed_channel

def unpad_block(padded_block, original_shape):
    return padded_block[:original_shape[0], :original_shape[1]]

def run_length_decode(rle_list, block_size=64):
    decoded = []
    
    for (zero_count, value) in rle_list:
        # Append the correct number of zeros followed by the non-zero value
        decoded.extend([0] * zero_count)
        decoded.append(value)
    
    # Ensure the decoded list has exactly block_size elements (e.g., 64 for an 8x8 block)
    decoded = decoded[:block_size]  # Truncate if necessary
    
    return decoded

# encoded_list - 3 tuples of values: huffman_encoded, huffman_tree
# Huffman Decoding → RLE Decoding → Inverse Zigzag Scan → Inverse Quantization → Inverse DCT → YCbCr to RGB
def decode(encoded_list, Q_list, block_size=8):
    height = 720  # original image height
    width = 1280  # original image width

    # Initialize decompressed channels for Y, Cb, and Cr
    decompressed_Y = np.zeros((height, width), dtype=np.float32)
    decompressed_Cb = np.zeros((height, width), dtype=np.float32)
    decompressed_Cr = np.zeros((height, width), dtype=np.float32)

    channels = [decompressed_Y, decompressed_Cb, decompressed_Cr]
    
    # Loop through the Y, Cb, and Cr channels
    for idx, (huffman_encoded_data, huffman_table) in enumerate(encoded_list):
        channel = huffman_encoded_data  # Get current channel to decode
        
        height, width = channel.shape
        
        # Step 1: Huffman Decode
        #channel = huffman_decode(channel, huffman_table)
        
        data_index = 0  # Initialize the index for RLE data
        
        # Step 2: Loop over image blocks to decode RLE
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                
                #block = channel[i:i + block_size, j:j + block_size]
                block = channel[i:i + block_size, j:j + block_size]
                #rle_block = run_length_decode(rle_block, block_size**2)

                # Step 3: Inverse Zigzag Scan
                block = reverse_zigzag(block, block_size)

                # Step 4: Assign decoded block back to channel
                channel[i:i + block_size, j:j + block_size] = block
                
                data_index += block_size ** 2
        
        # Step 5: Inverse Quantization and IDCT for the full channel
        channels[idx] = decode_PRIMARY(Q_list[idx], channel)
        channels[idx] = np.clip(channels[idx], 0, 255)  # Clip values to valid pixel range (0-255)

    # Return the decoded Y, Cb, Cr channels
    return np.dstack((channels[0], channels[1], channels[2])).astype(np.float32)

def decode2(Q, encoded_tuple, block_size=8):
    
    huffman_encoded_data, huffman_table = encoded_tuple
    
	# Step 1: Huffman Decode
    rle_data = huffman_decode(huffman_encoded_data, huffman_table)
    
    # Initialize variables for decompressed image (assume size is known or passed)
    height = 720  # original image height
    width = 1280   # original image width
    decompressed_channel = np.zeros((height, width), dtype=np.float32)
    
    data_index = 0
    
    # Step 2: Loop over image blocks to decode
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            rle_block = rle_data[data_index:data_index + block_size**2]
            data_index += block_size**2
            
            # Inverse Zigzag Scan and Dequantization
            dct_block = reverse_zigzag(rle_block, block_size)
            dequantized_block = dct_block * Q
            
            # Inverse DCT to get the original block
            spatial_block = idct(dequantized_block)
            
            # Place the block in the decompressed channel
            decompressed_channel[i:i + block_size, j:j + block_size] = spatial_block
    
    return decompressed_channel

    channel_array = channel[1]  # Assuming the second element is my 2D array

    # Convert to a NumPy array and ensure float type
    channel_array = np.array(channel_array, dtype=np.float64)

    # Check the shape of the channel_array
    if channel_array.shape != (8, 8):
        # Reshape or pad the array to fit (8, 8) if needed
        # Example of padding with zeros (adjust based on your actual data)
        padded_array = np.zeros((8, 8), dtype=np.float64)
        padded_array[:channel_array.shape[0], :channel_array.shape[1]] = channel_array
        channel_array = padded_array
        
    # Dequantization step: multiply by the quantization matrix Q
    dequantized_channel = channel_array * Q
    
    # Reverse zigzag (optional): If the channel was zigzagged, reverse it here
    block_size = 8
    reversed_zigzag_channel = reverse_zigzag(dequantized_channel, block_size)
    
    # Apply inverse DCT (IDCT) to each 8x8 block
    idct_channel = apply_idct(reversed_zigzag_channel, block_size)
    
    return idct_channel

# Peak Signal-to-Noise Ratio (PSNR). Measures quality of reconstruction. Higher PSNR values - better quality.
def PSNR(rgb_pixels, decoded_pixels):
	mse = mean((rgb_pixels - decoded_pixels) ** 2)

	if mse == 0:
		psnr = float('inf')  # Infinite PSNR for identical images
	else:
		psnr = 10 * math.log10((255 ** 2) / mse)
	return psnr

# Compression Ratio
def CR (rgb_pixels, output_filename):
	original_size = rgb_pixels.nbytes  # Size of original image
	compressed_size = os.path.getsize(output_filename)   # Size of compressed image
	
	CR = original_size / compressed_size
	return CR

def checkDetail(rgb_pixels): 
	# Convert the RGB image to grayscale
	gray_image = rgb2gray(rgb_pixels)

	# Entropy calculation (Shannon entropy)
	entropy_value = shannon_entropy(gray_image)
	print(f"\nImage Entropy: {entropy_value}")

#######################################################################
#Main program part
#######################################################################
#input_filename = "landscape.jpg" 
input_filename = "ocean 1280x720.jpg" 
#input_filename = "justblue.jpg" 
output_filename = "compressed_image.jpeg"

rgb_pixels = plt.imread(input_filename) # loads an image from a file into a NumPy array (rgb_pixels). The image is read in RGB format.
#checkDetail(rgb_pixels)

dimensions = rgb_pixels.shape[:-1] # Extracts the dimensions of the image excluding the color channels. shape returns a tuple where the last value is the number of channels.
print("\nShape in pixels: ", rgb_pixels.shape, ". Type: ", rgb_pixels.dtype)
#show_image(rgb_pixels)

YCbCr_pixels = rgb_to_YCbCr(rgb_pixels)

# https://stackoverflow.com/questions/74158097/how-to-display-colored-y-cb-cr-channels-in-python-opencv
# Creates an array filled with a constant value (128) to visualize the Y, Cb, and Cr channels.
grey_comp = full(dimensions, 128, uint8) 

# Visualize YCbCr channels
# When i == 0, comp refers to the Y channel, i == 1: Cb, i == 2: Cr
#for i, comp in enumerate(components(YCbCr_pixels)):
  #  show_image(visualize_channel(comp, i, grey_comp).astype(uint8))

sampleMode = 420
subsampled = subsample_YCbCr(YCbCr_pixels, sampleMode)

# Quantization Matrix
# https://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression
QF = clip(50, 1, 100) # Quantization factor between 1 and 100 - 1 very high compression (low quality), 100 - least compression (high quality)

# S is a scaling factor computed based on the quality factor QF. 
# It adjusts the quantization matrix's values, affecting how much compression is applied.
# For QF < 50: The scaling factor is calculated as 5000 / QF and quantization becomes more aggressive as QF decreases, leading to higher compression and lower quality.
# For QF >= 50: The formula 200 - (2 * QF) is applied, resulting in less aggressive quantization, meaning less compression and higher quality.
if QF < 50:
    S = 5000 / QF
else:
    S = 200 - (2 * QF)

# Divide by 100 to get the final scaling factor
S = S / 100

# Function to calculate the quantization matrices based on the scaling factor S
B = 8 # block size
QY, QC = calculate_quantization_matrices(S, B)

Q = (QY, QC, QC)
C = contruct_C(QY, B) # coefficient matrix C for DCT 

# Encodes each channel using DCT and quantization matrices.
#encoded = [compress_channel(channel, quant_matrix) for channel, quant_matrix in zip(subsampled, Q)]
#encoded = (compress_channel(*p) for p in zip(Q, subsampled))
encoded = [compress_channel(q, c) for q, c in zip(Q, subsampled)]
#encoded = (encode_PRIMARY(*p) for p in zip(Q, subsampled))

# After decoding all the channels (Y, Cb, and Cr), 
# it stacks them back together along the third dimension using dstack.
# The result is a full decoded image in the YCbCr color space.
#decoded = dstack(tuple(decode_PRIMARY(*p) for p in zip(Q, encoded)))

decoded = decode(encoded, Q)
#decoded = dstack(tuple(decode(q, e) for q, e in zip(Q, encoded)))
#decoded = dstack(tuple(decode(*p) for p in zip(Q, encoded)))

# Converts the decoded YCbCr image back to the RGB color space using 
# the YCbCr_to_rgb function. 
# The pixel values are cast to uint8 to ensure they're in a
#  correct format for displaying.
decoded_pixels = YCbCr_to_rgb(decoded).astype(uint8)

# should look like compressed version of the original picture
show_image(decoded_pixels)

# Convert the NumPy array (decoded_pixels) to a PIL Image object
compressed_image = Image.fromarray(decoded_pixels)
# Save the image as a JPEG with specified quality (you can adjust the quality as needed)
compressed_image.save(output_filename, format="JPEG")
# Optionally, print a confirmation message
print(f"Image saved as {output_filename}")

# Root Mean Squared Error (RMSE) between the original image (rgb_pixels) 
# and the decoded (reconstructed) image (decoded_pixels) using mean_squared_error. RMSE is a common metric to measure 
# the difference between two images — lower values indicate better quality.
RMSE1 = math.sqrt(mean_squared_error(rgb_pixels, decoded_pixels))
print("RMSE: ", RMSE1)

# Peak Signal-to-Noise Ratio (PSNR). Measures quality of reconstruction. Higher PSNR values - better quality.
PSNR1 = PSNR(rgb_pixels, decoded_pixels)
print("PSNR: ", PSNR1)

# Compression Ratio
CR = CR(rgb_pixels, output_filename)
print("CR: ", CR)

# %%
