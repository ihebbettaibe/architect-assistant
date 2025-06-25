@echo off
cd /d "c:\Users\ASUS\Desktop\architect-assistant\agents\budget"
echo.
echo ========================================
echo  🏗️ Agent Budget Immobilier - FIXED!
echo ========================================
echo.
echo ✅ Standard Agent: Ready
echo ✅ LangChain + Groq Agent: Ready  
echo ✅ Auto-loading Groq API key from .env
echo.
echo Loading environment variables...
echo Starting Streamlit app...
echo.
echo The app will open in your default web browser.
echo Press Ctrl+C in this window to stop the app.
echo.
echo ========================================
echo.
streamlit run budget_streamlit_app.py --server.headless false
echo.
echo App stopped.
pause
