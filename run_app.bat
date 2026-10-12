@echo off
echo Starting Salary Prediction App...
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
