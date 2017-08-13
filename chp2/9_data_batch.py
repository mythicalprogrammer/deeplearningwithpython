# first batch of 128
batch = train_images[:128]

# next batch (second batch)
batch = train_images[128:256]

# and the n-th batch:
batch = train_images[128 * n:128 * (n + 1)]

# start range 128*n 
# end range 128*(n+1)
