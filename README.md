# mibi-bin-tools

Toolbox for extracting tiff images from MIBIScope .bin files 

## To install the project:

Open terminal and navigate to where you want the code stored.

Then input the command:

```
git clone https://github.com/angelolab/mibi-bin-tools.git
```

Next, you'll need to set up a docker image with all of the required dependencies.
 - First, [download](https://hub.docker.com/?overlay=onboarding) docker desktop. 
 - Once it's sucessfully installed, make sure it is running by looking in toolbar for the Docker whale.
 - Once it's running, enter the following commands into terminal 

```
cd mibi-bin-tools
docker build -t mibi-bin-tools '.'
``` 

You've now installed the code base. 

## Whenever you want to run the scripts:

Enter the following command into terminal from the same directory you ran the above commands:

```
./start_docker.sh
``` 

This will generate a link to a jupyter notebook. Copy the last URL (the one with `127.0.0.1:8888` at the beginning) into your web browser.

Be sure to keep this terminal open.  **Do not exit the terminal or enter control-c until you are finished with the notebooks**.

### NOTE

If you already have a Jupyter session open when you run `./start_docker.sh`, you will receive a couple additional prompts. 

Copy the URL listed after `Enter this URL instead to access the notebooks:` 

You will need to authenticate. Note the last URL (the one with `127.0.0.1:8888` at the beginning), copy the token that appears there (it will be after `token=` in the URL), paste it into the password prompt of the Jupyter notebook, and log in.

## Using the example notebook(s):
- The ExtractBinFiles notebook provides an easy interface for getting tifs from bin files. 

## Once you are finished

You can shut down the notebooks and close docker by entering control-c in the terminal window.

## Updates

This project is still in development, and we are making frequent updates and improvements. If you want to update the version on your computer to have the latest changes, perform the following steps

First, get the latest version of the code

```
git pull
```

Then, run the command below to update the jupyter notebooks to the latest version
```
./start_docker.sh --update
```
or
```
./start_docker.sh -u
```

If the requirements.txt has changed, Docker will rebuild with the new dependencies first.

### WARNING

If you didn't change the name of any of the notebooks within the `scripts` folder, they will be overwritten by the command above!

If you have made changes to these notebooks that you would like to keep (specific file paths, settings, custom routines, etc), rename them before updating!

For example, rename your existing copy of `ExtractBinFile.ipynb` to `ExtractBinFile_old.ipynb`. Then, after running the update command, a new version of `ExtractBinFile.ipynb` will be created with the newest code, and your old copy will exist with the new name that you gave it. 

After updating, you can copy over any important paths or modifications from the old notebooks into the new notebook

## Questions?

If that doesn't answer your question, you can open an [issue](https://github.com/angelolab/mibi-bin-tools/issues). Before opening, please double check and see that someone else hasn't opened an issue for your question already. 
