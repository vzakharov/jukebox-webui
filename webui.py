import gradio as gr
import os
import urllib.request
import yaml

yaml.add_representer(str, lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|'))

# Define a data class
class Data:

  base_folder = 'G:\Мой диск\AI music'  #@param {type:"string"}
  project_name = ''                     #@param {type:"string"}
  artist = ''                           #@param {type:"string"}
  genre = ''                            #@param {type:"string"}
  lyrics = ''                           #@param {type:"string"}
  duration = 200                        #@param {type:"number"}

  prime_filename = ''                   #@param {type:"string"}
  zs_filename = ''                      #@param {type:"string"}

  generation_length = 3                 #@param {type:"number"}

# Create a data object
data = Data()

def get_project_names():
  project_names = []
  for folder in os.listdir(data.base_folder):
    if os.path.isdir(data.base_folder+'/'+folder):
      project_names.append(folder)
  # Sort project names alphabetically
  project_names.sort()
  return project_names

def get_list(name):
  items = []
  with urllib.request.urlopen(f'https://raw.githubusercontent.com/openai/jukebox/master/jukebox/data/ids/v2_{name}_ids.txt') as f:
    for line in f:
      item = line.decode('utf-8').split(';')[0]
      item = item.replace('_', ' ').title()
      items.append(item)
  items.sort()
  return items

# Create an Inputs class with the same attributes as Data but they refer to gradio inputs, not actual values.
# Each input should be None by default.
class Inputs:

  def __init__(self, data):

    for key in dir(data):
      if not key.startswith('__'):
        setattr(self, key, None)
    
    self.data = data  

# Create an inputs object
inputs = Inputs(data)

def to_snake_case(string):
  return string.replace(' ', '_').lower()

# Add a 'reactive' instance method to gradio's Changeable class
# Use global data object to sync values with
def reactive(self):

  global data, inputs

  key = to_snake_case(self.label)

  def set_value(value):
    setattr(data, key, value)
    print(f'{key} => {getattr(data, key)}')
    assert getattr(data, key) == value, f'Could not set {key} to {value}'
  
  self.change(
    fn = set_value,
    inputs = self,
    outputs = [],
  )

  self.value = getattr(data, key)
  setattr(inputs, key, self)

  return self

from gradio.events import Changeable
Changeable.reactive = reactive

# Function to get all files with the extension '.zs' in the project folder
def get_zs_files(base_folder, project_name):
  zs_files = []
  for file in os.listdir(f'{base_folder}/{project_name}'):
    if file.endswith('.zs'):
      zs_files.append(file)
  # Sort zs files alphabetically in reverse order
  zs_files.sort(reverse=True)
  return zs_files

def update_zs_files(base_folder, project_name):
  new_choices = get_zs_files(base_folder, project_name)
  return gr.Dropdown.update(
    choices = new_choices,
  )


with gr.Blocks() as ui:

  with gr.Row():

    with gr.Column(scale=1):

      gr.Textbox(label='Base folder').reactive()
      gr.Dropdown(label='Project name', choices=get_project_names()).reactive()

      gr.Dropdown(label='Artist', choices=get_list('artist')).reactive()
      gr.Dropdown(label='Genre', choices=get_list('genre')).reactive()
      gr.Textbox(label='Lyrics', max_lines=10).reactive()
      gr.Slider(label='Duration', minimum=60, maximum=600, step=10).reactive()


      # Button to load data from settings.yaml in the project folder
      def load_settings(base_folder, project_name):
        filename = f'{base_folder}/{project_name}/settings.yaml'
        if os.path.exists(filename):
          with open(filename, 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
            print(f'Loaded settings from {filename}:')
            print(settings)
            for key in settings:
              setattr(data, key, settings[key])
        return [ data.artist, data.genre, data.lyrics, data.duration ]

      gr.Button('Load settings').click(load_settings, 
        inputs=[ inputs.base_folder, inputs.project_name ], 
        outputs=[ inputs.artist, inputs.genre, inputs.lyrics, inputs.duration ]
      )

      # Button to save project settings to settings.yaml in the project folder
      def save_settings(base_folder, project_name, artist, genre, lyrics, duration):
        filename = f'{base_folder}/{project_name}/settings.yaml'
        settings = {
          'artist': artist,
          'genre': genre,
          'lyrics': lyrics,
          'duration': duration
        }
        with open(filename, 'w') as f:
          yaml.dump(settings, f)
        print(f'Saved settings to {filename}:')
        print(settings)
      
      gr.Button('Save settings').click(save_settings,
        inputs=[ inputs.base_folder, inputs.project_name, inputs.artist, inputs.genre, inputs.lyrics, inputs.duration ],
        outputs=[]
      )
    
    with gr.Column(scale=3):

      with gr.Tab('Start'):
        # (Tab for initial sample generation)

        # Browse for prime wav file
        inputs.prime_filename = gr.Audio(label='Prime file', interactive=True)

      with gr.Tab('Continue'):
        # (Tab for continuation of existing sample)
        
        # Dropdown to select a zs file
        inputs.zs_filename = gr.Dropdown(label='ZS file', interactive=True)

        # Whenever the base folder or project name changes, update the list of zs files
        project_defining_inputs = [inputs.base_folder, inputs.project_name]

        for input in project_defining_inputs:
          input.change(
            fn = update_zs_files,
            inputs = project_defining_inputs,
            outputs = inputs.zs_filename,
          )



      inputs.generation_length = gr.Slider(label='Generation length (seconds)', minimum=1, maximum=30, value=data.generation_length, step=1)

      # Breakpoint button
      def _breakpoint():
        global inputs, data
        breakpoint()

      gr.Button('Breakpoint').click(_breakpoint, inputs=[], outputs=[])


ui.launch(share=True)