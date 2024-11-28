export COMMAND="alias evrun='python $(pwd)/bin/es.py'"
if [ -f ~/.bash_aliases ]; then
    if ! grep -Fxq "$COMMAND" ~/.bash_aliases; then
        echo $COMMAND >> ~/.bash_aliases
    fi
else
   echo $COMMAND > ~/.bash_aliases
fi