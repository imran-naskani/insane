Set-Location "C:\Users\m_imr\Downloads\INSANE"

git add .
git commit -m "Auto update $(Get-Date -Format 'yyyy-MM-dd HH:mm')" 2>$null
git push origin main
