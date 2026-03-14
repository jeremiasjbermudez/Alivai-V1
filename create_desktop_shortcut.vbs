Set oWS = WScript.CreateObject("WScript.Shell")
strDesktop = oWS.SpecialFolders("Desktop")
set oShortcut = oWS.CreateShortcut(strDesktop & "\Open Alivai Dashboard & Logs.lnk")
oShortcut.TargetPath = "c:\Users\Alivai\Documents\Alivai_V1\open_dashboard_and_logs.bat"
oShortcut.WorkingDirectory = "c:\Users\Alivai\Documents\Alivai_V1"
oShortcut.WindowStyle = 1
oShortcut.Save
