from PIL import Image


def convert_jpeg_to_bmp(jpeg_file, bmp_file):
    # Open the JPEG file
    with Image.open(jpeg_file) as img:
        # Save as BMP
        img.save(bmp_file, "BMP")


def convert_bmp_to_jpeg(bmp_file, jpeg_file):
    # Open the BMP file
    with Image.open(bmp_file) as img:
        # Convert and save as JPEG
        img.convert("RGB").save(jpeg_file, "JPEG")


# Comment or uncoment this
convert_jpeg_to_bmp("sample_face.jpg", "sample_face.bmp")
