# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.message import EmailMessage

# Open the plain text file whose name is in textfile for reading.
msg = EmailMessage()
msg.set_content('test')

# me == the sender's email address
# you == the recipient's email address
msg['Subject'] = f'The contents'
msg['From'] = 'rico.shin@gmail.com'
msg['To'] = 'rico.shin@gmail.com'

# Send the message via our own SMTP server.
s = smtplib.SMTP('smtp.gmail.com, 587')
s.starttls()
s.login('rico.shin@gmail.com', 'iigxbakkrpzduwtm')
s.send_message(msg)
s.quit()
