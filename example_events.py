import json

events = [
    {
        "body": json.dumps(
            {
                "document_text": """Lufthansa flies back to profit
                            
                            German airline Lufthansa has returned to profit in 2004 after posting huge losses in 2003.
                            
                            In a preliminary report, the airline announced net profits of 400m euros ($527.61m; £274.73m), compared with a loss of 984m euros in 2003. Operating profits were at 380m euros, ten times more than in 2003. Lufthansa was hit in 2003 by tough competition and a dip in demand following the Iraq war and the killer SARS virus. It was also hit by troubles at its US catering business. Last year, Lufthansa showed signs of recovery even as some European and US airlines were teetering on the brink of bankruptcy. The board of Lufthansa has recommended paying a 2004 dividend of 0.30 euros per share. In 2003, shareholders did not get a dividend. The company said that it will give all the details of its 2004 results on 23 March.
                            """
            }
        )
    }
]
