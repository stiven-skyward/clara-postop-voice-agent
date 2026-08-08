# Ejecutar en PowerShell COMO ADMINISTRADOR en Windows.
# Reenvía localhost:8000 de Windows al servidor dentro de WSL2.
# Necesario solo si http://localhost:8000 no abre desde el navegador de Windows
# (el reenvío automático de WSL2 a veces falla). La IP de WSL cambia en cada
# reinicio: vuelve a ejecutar este script si eso pasa.

$ip = (wsl hostname -I).Trim().Split()[0]
if (-not $ip) { Write-Error "No se pudo obtener la IP de WSL"; exit 1 }

netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=127.0.0.1 2>$null | Out-Null
netsh interface portproxy delete v6tov4 listenport=8000 listenaddress=::1     2>$null | Out-Null
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=127.0.0.1 connectport=8000 connectaddress=$ip
netsh interface portproxy add v6tov4 listenport=8000 listenaddress=::1       connectport=8000 connectaddress=$ip

Write-Host "Reenvio configurado hacia $ip :"
netsh interface portproxy show all
Write-Host "`nAbre http://localhost:8000 (el microfono funciona: localhost es origen seguro)."
