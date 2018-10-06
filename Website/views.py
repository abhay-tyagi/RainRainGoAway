from django.shortcuts import render
from .models import Vid
from .utilities import testing
import os

# Create your views here.

def index(request):
	if request.method == 'GET':
		return render(request, 'index.html', {'flag': 1})
	else:

		invid = request.FILES['invid']

		print(invid)

		vid = Vid()
		vid.inputVid = invid
		vid.save()

		oname = str(vid.inputVid)

		testing.improveVideo(str(vid.inputVid))

		vid.outputVid = 'output_' + oname.split('.')[0] + '.avi'
		
		nm = 'output_' + oname.split('.')[0]
		
		os.system("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = 'RainRemove/media/' + str(vid.outputVid), output = 'RainRemove/media/' + nm))

		vid.outputVid = nm + '.mp4'
		vid.save()

		return render(request, 'index.html', {'flag': 2, 'vid': vid})
