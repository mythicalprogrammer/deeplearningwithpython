# train_images[i] is selecting specific elements in a tensor is called
# tensor slicing

# selecting digit from 10 to 100
my_slice = train_images[10:100]
print(my_slice.shape) # 90, 28, 28 we have 90 element that are 28x28

# example of selecting all images but 14x14 bottom right corner
my_slice = train_images[:, 14:, 14:] 