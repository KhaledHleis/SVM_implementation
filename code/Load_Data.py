import numpy as np




def load_mnist_images(filename):
    """
    load_mnist_images reads the MNIST images from the IDX file and returns
    a 28x28x[number of MNIST images] array containing the raw MNIST images.
    
    Parameters:
        filename (str): Path to the IDX file containing MNIST images.
        
    Returns:
        images (np.ndarray): Array of shape (num_images, 28, 28) containing the images.
    """
    with open(filename, 'rb') as f:
        # Read magic number
        magic = int.from_bytes(f.read(4), byteorder='big')
        assert magic == 2051, f"Bad magic number in {filename}"

        # Read dimensions
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)

    # Convert to float32 and normalize to [0,1]
    images = images.astype(np.float32) / 255.0

    return images




def load_mnist_labels(filename):
    """
    load_mnist_labels reads the MNIST labels from the IDX file and returns
    a 1D numpy array containing the labels.
    
    Parameters:
        filename (str): Path to the IDX file containing MNIST labels.
        
    Returns:
        labels (np.ndarray): Array of shape (num_labels,) containing the labels.
    """
    with open(filename, 'rb') as f:
        # Read magic number
        magic = int.from_bytes(f.read(4), byteorder='big')
        assert magic == 2049, f"Bad magic number in {filename}"

        # Read the number of labels
        num_labels = int.from_bytes(f.read(4), byteorder='big')

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        assert labels.shape[0] == num_labels, "Mismatch in label count"

    return labels

def Load_Data_mnist(image_file = "../given/train-images-idx3-ubyte_",label_file = "../given/train-labels-idx1-ubyte_"):
    # Load the MNIST dataset
    images = load_mnist_images(image_file)
    labels = load_mnist_labels(label_file)

    # Filter dataset to include only two classes (e.g., "0" and "1")
    # You may change the selected classes as needed
    selected_classes = [0, 1]
    selected_indices = np.isin(labels, selected_classes)
    original_X = images[selected_indices]
    y = labels[selected_indices]

    # Convert labels to binary format (-1 and 1)
    y = np.where(y == selected_classes[0], -1, 1)

    # Reshape images from 28x28 to 784 for each image, so X becomes (num_samples, 784)
    X = original_X.reshape(original_X.shape[0], -1).T  # Transpose to get (784, num_samples) for compatibility with solver
    X = X / 255.0

    print("y of shape:",y.shape)
    print("X of shape:",X.shape)
    return X,y
