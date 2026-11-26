$sourceDir = Join-Path $PSScriptRoot 'source'
$oldFile = Join-Path $sourceDir 'reference.doc'
$newFile = Join-Path $sourceDir 'reference.docx'
if ((Get-FileHash -LiteralPath $oldFile -Algorithm SHA256).Hash.ToLower() -ne '783d298e7090bab123b760d1cd6389ebdc32896ab96740bd81dcb7f6d2cd33a2') { throw 'Unexpected public attachment' }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
try {
    $document = $word.Documents.Open($oldFile, $false, $true)
    $document.SaveAs2($newFile, 16)
    $document.Close(0)
} finally {
    $word.Quit()
}
