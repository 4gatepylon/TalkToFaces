# NOTE make sure to run this on a cloud instance with a GPU, save the image, and then scp it down to your machine to view (or whatever)
import sda

print("Instantiating Video Animator")
va = sda.VideoAnimator(gpu=0)  # Instantiate the animator
print("Running...")
vid, aud = va("sample_face.bmp", "sample_audio.wav")
print("Saving video!")
va.save_video(vid, aud, "generated.mp4")
print("Done (in generated.mp4)")
