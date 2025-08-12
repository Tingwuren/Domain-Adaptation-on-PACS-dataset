import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# Check if running in Google Colab
try:
    from google.colab import output
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

# SHOW SOME RANDOM IMAGES
def show_random_images(dataset, n=5, mean=None, std=None):
	for i in range(n):
		j = np.random.randint(0, len(dataset))
		print('Label:',dataset[j][1])
		imgshow(dataset[j][0], mean=mean, std=std)

	return

def imgshow(img, mean=None, std=None):
	if mean == None or std == None:
		# use (0.5 0.5 0.5) (0.5 0.5 0.5) as mean and std
		img = img / 2 + 0.5     # unnormalize
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()
		# raise RuntimeError("You should pass mean and std to 'imgshow' method")
	else : 
		# use custom mean and std computed on the images
		mean = np.array(mean)
		std = np.array(std)
		for i in range(3): 
			img[i] = img[i]*std[i] + mean[i] # unnormalize

		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()
	return

def plotLosses(class_loss, source_loss, target_loss, n_epochs=30, output_dir=None, show=False) : 
	epochs = range(n_epochs)
	plt.figure(figsize=(12, 8))
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	
	# Always plot classifier loss
	plt.plot(epochs, class_loss, 'b--', label="classifier", linewidth=2)
	
	# Only plot domain losses if they exist and have data
	if source_loss and len(source_loss) > 0:
		plt.plot(epochs, source_loss, 'g--', label="discriminator source", linewidth=2)
	
	if target_loss and len(target_loss) > 0:
		plt.plot(epochs, target_loss, 'r--', label="discriminator target", linewidth=2)
	
	plt.legend()
	plt.title('Training Losses Over Time')
	plt.grid(True, alpha=0.3)
	
	# Save to output directory if provided, otherwise current directory
	if output_dir:
		save_path = os.path.join(output_dir, 'losses.png')
		os.makedirs(output_dir, exist_ok=True)
	else:
		save_path = 'losses.png'
		
	plt.savefig(save_path, dpi=250, bbox_inches='tight')
	
	if show: 
		plt.show()
	else:
		plt.close()  # Close figure to free memory
		
	return save_path

def plotImageDistribution(data1, data2, data3, data4, dataset_names, classes_names, show=False):
	# concatenate datasets
	data = np.concatenate( (data1, data2, data3, data4) )
	# count element per class
	unique, counts = np.unique(data, return_counts=True)
	# for each domain
	unique, counts1 = np.unique(data1, return_counts=True)
	unique, counts2 = np.unique(data2, return_counts=True)
	unique, counts3 = np.unique(data3, return_counts=True)
	unique, counts4 = np.unique(data4, return_counts=True)

	if show: 
		print("------ Some statistics ------")
		print('Total images:', np.sum(counts))
		print('Number of classes:', len(unique))
		print('Classes:', unique)
		print('Classes Names:', classes_names) 
		print()
		print('Total images per class:', counts)
		print('Mean images per class:', counts.mean())
		print('Std images per class:', counts.std())
		print()
		print('Total images per domain/dataset:')
		print(f"Photo Dataset: {len(data1)}")
		print(f"Art Dataset: {len(data2)}")
		print(f"Cartoon Dataset: {len(data3)}")
		print(f"Sketch Dataset: {len(data4)}")
		print()
		print('Element per class for each domain:')
		for name,count in zip(dataset_names,[counts1,counts2,counts3,counts4]) : 
			print(f'{name}_dataset: {count}')

	fig, ax = plt.subplots(figsize=(10,7))

	width=0.18

	plt.bar(unique-2*(width)+(width/2), counts1, width=width, color='#FF8F77', linewidth=0.5, label='Photo')
	plt.bar(unique-(width/2), counts2, width=width, color='#FFDF77', linewidth=0.5, label='Art paintings')
	plt.bar(unique+(width/2), counts3, width=width, color='#8DF475', linewidth=0.5, label='Cartoon')
	plt.bar(unique+2*(width)-(width/2), counts4, width=width, color='#77DCFF', linewidth=0.5, label='Sketch')

	ax.set_xticks(unique)
	classes = ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person']
	ax.set_xticklabels(classes)

	plt.grid(alpha=0.2, axis='y')

	plt.legend()
	if show: 
		plt.show()
	plt.savefig('distribution.png', dpi = 250)
	return

def beep():
	# Play an audio beep. Any audio URL will do.
	if COLAB_AVAILABLE:
		output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')
	else:
		print("Beep! (Audio only available in Google Colab)")
	return 
