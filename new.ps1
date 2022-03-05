$filename = Get-Date -UFormat "posts/%y/%m/$args.md" 
hugo new $filename