echo OFF
echo[
echo[
echo                   ############################
echo                   ##   Building Word2Vec    ##
echo                   ############################
echo[
echo[

dotnet build -c Release

echo Word2Vec was built in release configuration, ready to create package
pause
echo[
echo[
echo                 ###################################
echo                 ##   Building Word2Vec Package   ##
echo                 ###################################
echo[
echo[

nuget pack Proxem.Word2Vec/nuspec/Proxem.Word2Vec.nuspec -Symbols -OutputDirectory Proxem.Word2Vec/nuspec/

echo Package was created in folder Proxem.Word2Vec/nuspec/
pause