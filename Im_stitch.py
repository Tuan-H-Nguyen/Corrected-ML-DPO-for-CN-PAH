#%%
from PIL import Image

def merge_image2(file1, file2):
    #Merge two images into one, displayed side by side
    #:param file1: path to first image file
    #:param file2: path to second image file
    #:return: the merged Image object
 
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = max(width1,width2)
    result_height = height1 + height2

    result = Image.new('RGB', (result_width, result_height),color = 'white')
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))
    return result

def side_merge_image2(file1, file2):
    #Merge two images into one, displayed side by side
    #:param file1: path to first image file
    #:param file2: path to second image file
    #:return: the merged Image object
 
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1,height2)

    result = Image.new('RGB', (result_width, result_height),color = 'white')
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result

def merge_image3(file1, file2,file3):
	image1 = Image.open(file1)
	image2 = Image.open(file2)
	image3 = Image.open(file3)
	
	(width1, height1) = image1.size
	(width2, height2) = image2.size
	(width3, height3) = image3.size
	
	result_width = max(width1,width2,width3)
	result_height = height1 + height2 + height3
	
	result = Image.new('RGB', (result_width, result_height),color = 'white')
	result.paste(im=image1, box=(0, 0))
	result.paste(im=image2, box=(0, height1))
	result.paste(im=image3, box=(0, height1 + height2))
	return result


merge_image3(
    "shifted_linear_train_BG.jpeg",
    "shifted_linear_train_EA.jpeg",
    "shifted_linear_train_IP.jpeg"
).save("shifted_linear_train.jpeg")


merge_image3(
    "BG_excl_DPO_vs_SP.jpeg",
    "EA_excl_DPO_vs_SP.jpeg",
    "IP_excl_DPO_vs_SP.jpeg"
).save("PROP_excl_DPO_vs_SP.jpeg")

merge_image3(
    "BG_excl_SP_vs_DPO.jpeg",
    "EA_excl_SP_vs_DPO.jpeg",
    "IP_excl_SP_vs_DPO.jpeg"
).save("PROP_excl_SP_vs_DPO.jpeg")

merge_image3(
    "test_BG.jpeg",
    "test_EA.jpeg",
    "test_IP.jpeg"
).save("test_.jpeg")


merge_image3(
    "manual_model_BG.jpeg",
    "manual_model_EA.jpeg",
    "manual_model_IP.jpeg"
).save("manual_model.jpeg")


merge_image3(
    "test_manual_model_BG.jpeg",
    "test_manual_model_EA.jpeg",
    "test_manual_model_IP.jpeg"
).save("test_manual_model.jpeg")

merge_image3(
    "_rmsd_vs_train_size_BG.jpeg",
    "_rmsd_vs_train_size_EA.jpeg",
    "_rmsd_vs_train_size_IP.jpeg"
).save("_rmsd_vs_train_size.jpeg")

# %%
