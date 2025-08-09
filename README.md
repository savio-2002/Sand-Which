<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />


# Sand-Which? 🎯


## Basic Details
### Team Name: ChaatGPT


### Team Members
- Team Lead: Manjith M - CUSAT
- Member 1: Savio Joemon- CUSAT
- Member 2: Manjith M - CUSAT

### Project Description
Sand-Which? is a quirky computer vision app that analyzes your half-eaten sandwich image using OpenCV and smart image processing, then suggests the most satisfying next bite. Because making the wrong choice is a recipe for regret. 🥪

### The Problem
Choosing the next bite of a sandwich is a high-stakes decision… that absolutely doesn’t need solving. Yet here we are. Why waste time thinking about calories when you can waste time thinking about the next bite?

### The Solution
We built Sand-Which?, a playful tool that scans your half-eaten sandwich and highlights the optimal next bite. No artificial intelligence, just some image processing, a bit of math, and a lot of overthinking your lunch.

## Technical Details
### Technologies/Components Used
For Software:
- Python
- Streamlit
- OpenCV (image processing), NumPy, Pillow
- GitHub (version control), Pinterest (for sandwich images)

For Hardware:
- None required beyond a device with a web browser and internet connection
- Optional: A sandwich (for field testing)
- Camera or phone to take sandwich pictures

## Implementation
For Software:
### Clone the repo  
```
git clone [https://github.com/tinkerhub/Sand-Which.git  ](https://github.com/savio-2002/Sand-Which.git)
cd Sand-Which  
python -m venv venv  

### Windows  
venv\Scripts\activate

### macOS/Linux  
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

### Project Documentation
For Software:

Flow:

User uploads a photo of a partially eaten sandwich

The app processes the image, finds edges, and locates the “next best bite” zone

Displays a zoomed-in view of the bite target with a fun UI background

Features:

Dark-themed foreground container for contrast

Background image fetched from Pinterest (because why not?)

“Risky Bite Mode” for those who like danger in their sandwiches

Option to waste more time thinking about bites than calories

Limitations:

Not AI-powered — but your sandwich will believe it is

Works best if you don’t eat the entire sandwich before uploading

# Screenshots
<img width="1920" height="1080" alt="Screenshot 2025-08-09 154837" src="https://github.com/user-attachments/assets/2557a894-5594-411d-9c12-5b817f618328" /><br>
<p align = "center">Home Page</p>
<br><img width="815" height="1044" alt="Screenshot 2025-08-09 155002" src="https://github.com/user-attachments/assets/cfc7c207-0c52-4976-b267-3ed972f1ae29" /><br>
<p align = "center">After user uploads photo</p>
<br><img width="941" height="997" alt="Screenshot 2025-08-09 155053" src="https://github.com/user-attachments/assets/70905422-c8dc-4bf5-a92c-895aba049d25" /><br>
<p align = "center">Final Result showing the best bite</p>
<br><img width="868" height="990" alt="Screenshot 2025-08-09 160059" src="https://github.com/user-attachments/assets/9ec018fe-ec75-4d62-8bb8-e584d449a7fe" /><br>
<p align = "center">Random Bite Result</p>


## Team Contributions
- Savio Joemon: Developed the core image processing workflow (bite prediction, highlighting areas).
                Integrated user options such as “Risky Bite Mode” with unique Streamlit widget keys.
                Set up installation instructions, documentation, and GitHub project structure.

- Manjith M: Designed the UI and branding for Sand-Which? including color schemes, and layout.
             Implemented image upload, bite-detection logic, and result display.
             Handled styling with custom CSS for the foreground container and overall look.


---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)



