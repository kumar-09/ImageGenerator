This is a project which will generate a random image using and AI model upon refresh of page.
The frontend is made of React
There are two server used--
  one where AI model is placed, it will only hadle the AI part of the backend. I have made it using Flask and touch
  other from where we will just call this AI model server which will give the random image to display on frontend.It is based on Django
  Since the AI model take to much space to run so we have putted it on different server so that it will not cause any disturbance in other API request coming from frontend to main backend server
  
  
