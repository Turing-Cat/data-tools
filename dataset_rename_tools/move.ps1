# 将子目录中的图像和标注移动到当前目录
$baseDir = "D:\Datasets\DATASET-Label\single-object\Labeled\2"
Get-ChildItem -Path $baseDir -Directory | ForEach-Object {
    $subDir = $_.FullName
    Get-ChildItem -Path $subDir -File | Where-Object { $_.Name -notlike "*.json" } | Copy-Item -Destination $baseDir
}