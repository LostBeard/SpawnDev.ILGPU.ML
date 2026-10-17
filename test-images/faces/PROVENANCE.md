# Test faces (public domain)

Official portraits of US government officials, Wikimedia Commons, licence "Public domain" per each file's extmetadata (queried 2026-10-15).
Stored here as 360 px-wide raw 32-bit B, G, R, A (`*.bgra`), made by `oracle_faces.py` (see `faces-fixture.json`: what OpenCV 5.0 FaceDetectorYN + FaceRecognizerSF return for each).

| file | Commons title |
|---|---|
| biden_2013.jpg | File:Joe_Biden_official_portrait_2013.jpg |
| biden_2021.jpg | File:Joe_Biden_presidential_portrait.jpg |
| trump_2017.jpg | File:Donald_Trump_official_portrait.jpg |
| trump_2025.jpg | File:Official_Presidential_Portrait_of_President_Donald_J._Trump_(2025).jpg |
| harris_2021.jpg | File:Kamala_Harris_Vice_Presidential_Portrait.jpg |
| pence_2017.jpg | File:Mike_Pence_official_Vice_Presidential_portrait.jpg |

Used only as fixtures for face detection / recognition parity and identity-separation tests (two photos of two people, plus two single-photo distractors).
