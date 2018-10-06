from django.shortcuts import render
from .models import Vid
# from .utilities import testing

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

		# testing.improveVideo('media/' + vid.inputVid)

		return render(request, 'index.html', {'flag': 2, 'vid': vid})