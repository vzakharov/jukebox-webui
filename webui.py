import gradio as gr
import json
import os
import glob
import urllib.request
import yaml
import re

# Dump strings as literal blocks in YAML
yaml.add_representer(str, lambda dumper, value: dumper.represent_scalar('tag:yaml.org,2002:str', value, style='|'))

def get_project_names(base_folder):
  project_names = []
  for folder in os.listdir(base_folder):
    if os.path.isdir(base_folder+'/'+folder) and not folder.startswith('.'):
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

def get_files(path, extension):
  files = []
  for file in os.listdir(path):
    if file.endswith(extension):
      files.append(file)
  # Sort backwards
  files.sort(reverse=True)
  return files

def update_list(extension):
  def update_function(base_folder, project_name):
    print(f'Updating {extension} list for {project_name}...')
    return gr.update(
      choices = get_files(f'{base_folder}/{project_name}', extension),
    )
  print(f'Created update function for {extension}')
  return update_function

def update_samples(base_folder, project_name):
  print(f'Updating samples list for {project_name}...')
  # If the filename is 'project-part1-1-3-2-4,5,6.zs', sample ids will be 'part1-1-3-2-4', 'part1-1-3-2-5', and 'part1-1-3-2-6'
  sample_ids = []
  for file in os.listdir(f'{base_folder}/{project_name}'):
    if file.endswith('.zs'):
      # Remove path, project name, and extension
      full_id = file.replace(f'{project_name}-', '').replace('.zs', '')
      parts = full_id.split('-')
      sample_ids += [ 
        '-'.join(parts[:-1] + [id]) 
          for id in parts[-1].split(',') 
      ]
  # Reverse sort
  sample_ids.sort(reverse=True)
  return gr.update(
    choices = sample_ids,
  )

def update_project_data(base_folder, project_name):

  out_dict = {}

  # Load data from settings.yaml in the project folder, if it exists
  data_filename = f'{base_folder}/{project_name}/settings.yaml'
  if os.path.exists(data_filename):
    with open(data_filename, 'r') as f:
      data = yaml.load(f, Loader=yaml.FullLoader)
      print(f'Loaded settings from {data_filename}:')
      print(data)
      for key in data:
        out_dict[getattr(UI, key)] = data[key]

  return out_dict

def save_project_data(base_folder, project_name, artist, genre, lyrics, duration, step, sample_id, prime_id, generation_length):
  # Dump all arguments except base_folder and project_name
  data = {key: value for key, value in locals().items() if key not in ['base_folder', 'project_name']}
  filename = f'{base_folder}/{project_name}/settings.yaml'
  with open(filename, 'w') as f:
    yaml.dump(data, f)
    # print(f'Settings updated.')
    # print(data) 

class UI:

  # Define UI elements

  base_folder = gr.Textbox(label='Base folder')
  project_name = gr.Dropdown(label='Project name')

  create_project_button = gr.Button('Create')

  artist = gr.Dropdown(label='Artist', choices=get_list('artist'))
  genre = gr.Dropdown(label='Genre', choices=get_list('genre'))
  lyrics = gr.Textbox(label='Lyrics', max_lines=8)
  duration = gr.Slider(label='Duration', minimum=60, maximum=600, step=10)

  step = gr.Radio(label='Step', choices=['Start', 'Continue'])

  prime_id = gr.Dropdown(label='Prime Id')
  sample_id = gr.Dropdown(label='Sample Id')

  generation_length = gr.Slider(label='Generation length', minimum=1, maximum=10, step=0.25)

  sample_audio = gr.Audio(visible=False)
  sample_children_audios = [ gr.Audio(visible=False) for i in range(10) ]

  all_inputs = [ input for input in locals().values() if isinstance(input, gr.Interface) ]

  project_defining_inputs = [ base_folder, project_name ]
  project_specific_inputs = [ artist, genre, lyrics, duration ]
  generation_specific_inputs = [ step, sample_id, prime_id, generation_length ]

  general_inputs = project_defining_inputs

  with gr.Blocks() as ui:

    with gr.Row():

      with gr.Column(scale=1):

        with gr.Box():
          
          base_folder.render()

          # Radio buttons to either open an existing project or create a new one
          open_or_create_project = gr.Radio(
            label='Project',
            choices=['Open', 'New'],
            value='Open'
          )

          with gr.Box(visible=False) as open_project_box:
            # Open a project
            project_name.render()
          
          with gr.Box(visible=False) as create_project_box:
            # Create a new project

            new_project_name = gr.Textbox(label='New project name')

            def create_new_project(base_folder, new_project_name):
              print(f'Creating new project {new_project_name}...')
              path = f'{base_folder}/{new_project_name}'
              # If the project folder already exists, throw an error
              assert not os.path.exists(path), f'Project folder {path} already exists!'
              os.mkdir(f'{base_folder}/{new_project_name}')
              return {
                UI.project_name: gr.update(
                  choices = get_project_names(base_folder),
                  value = new_project_name,
                ),
                UI.open_project_box: gr.update(visible=True),
                UI.create_project_box: gr.update(visible=False),
                UI.open_or_create_project: 'Open'
              }

            create_project_button.render()

            create_project_button.click(
              inputs = [ base_folder, new_project_name ],
              outputs = [ project_name, open_project_box, create_project_box, open_or_create_project ],
              fn = create_new_project,
            )

          open_or_create_project.change(
            inputs = open_or_create_project,
            outputs = [ open_project_box, create_project_box ],
            fn = lambda action: [
              gr.update(visible=a==action) for a in ['Open', 'New']
            ]
          )
        
        with gr.Box(visible=False) as project_box:
          
          [ input.render() for input in project_specific_inputs ]

      with gr.Column(scale=3, visible=False) as generation_box:

        step.render()

        with gr.Box(visible=False) as start_box:
          prime_id.render()

        with gr.Box(visible=False) as continue_box:
          sample_id.render()

          # Button to go to parent sample
          def go_to_parent(base_folder, project_name, sample_id):
            parent_id = re.sub(r'-\d+$', '', sample_id)
            assert parent_id != sample_id, 'Can’t deduce parent id from sample id'
            # Assert that parent id exists (there is a wav file for it)
            assert os.path.exists(f'{base_folder}/{project_name}/{project_name}-{parent_id}.wav'), 'Parent id does not exist'
            return parent_id

          gr.Button('<< To parent sample').click(
            go_to_parent,
            inputs = [ base_folder, project_name, sample_id ],
            outputs = sample_id,
          )
          
        generation_length.render()
        sample_audio.render()

        gr.Markdown('**Children**')
        
        [ audio.render() for audio in sample_children_audios ]

    # Event logic

    # If base_folder changes, update available project names
    base_folder.submit(
      lambda base_folder: gr.update(choices=get_project_names(base_folder)),
      inputs=base_folder,
      outputs=project_name,
    )

    # If project_name changes, make project and generation boxes visible
    project_name.change(
      lambda: [ gr.update(visible=True), gr.update(visible=True) ],
      inputs = None,
      outputs = [ project_box, generation_box ],
    )

    # Whenever the base folder or project name changes, update the project data and save general inputs to settings.json (in root folder)
    for input in project_defining_inputs:

      # Update input values
      input.change(update_project_data, 
        inputs = project_defining_inputs,
        outputs = project_specific_inputs + generation_specific_inputs,
      )

      # Update prime file choices
      input.change(update_list('.wav'), 
        inputs = project_defining_inputs,
        outputs = prime_id,
      )

      # Update sample choices
      input.change(update_samples,
        inputs = project_defining_inputs,
        outputs = sample_id,
      )

      # Write general data to settings.json
      def save_general_data(*args):
        print('Saving general data')
        data = {}
        items = UI.__dict__.items()
        print(f'Items: {items}')
        for i, input in enumerate(UI.general_inputs):
          print(f'Input {i}: {input}')
          name = [ key for key, value in items if value == input ][0]
          value = args[i]
          print(f'{name}: {value}')
          data[name] = value
        with open('settings.json', 'w') as f:
          json.dump(data, f)
      
      input.change(save_general_data,
        inputs = general_inputs,
        outputs = None,
      )      


    # Whenever a project-specific input changes, save the project data
    for input in project_specific_inputs + generation_specific_inputs:
      input.change(save_project_data, 
        inputs = project_defining_inputs + project_specific_inputs + generation_specific_inputs,
        outputs = [],
      )
    
    step.change(
      lambda step: [ gr.update(visible=step==current_step) for current_step in ['Start', 'Continue'] ],
      inputs=step, outputs=[start_box, continue_box]
    )      

    # When the sample changes, unhide if [project_name]-[sample_id].wav exists and point it to that file
    def update_sample_wavs(base_folder, project_name, sample_id):
      filename = f'{base_folder}/{project_name}/{project_name}-{sample_id}.wav'
      print(f'Looking for {filename}')
      if os.path.exists(filename):
        print(f'Found {filename}')
        children = []
        for file in glob.glob(f'{base_folder}/{project_name}/{project_name}-{sample_id}-*.wav'):
          # Remove project name and extension
          match = re.match(f'.*{project_name}-{sample_id}-(\d+)\.wav', file)
          if not match:
            continue
          child_id = match.group(1)
          if child_id:
            print(f'Found child {child_id}')
            children.append(
              gr.update(
                visible=True,
                value=file,
                label=f'{sample_id}-{child_id}',
              )
            )
            # If already at max children, stop
            if len(children) == 10:
              break
        
        # If length is less than 10, hide the rest
        for i in range(len(children), 10):
          children.append(
            gr.update(visible=False)
          )

        return [ gr.update(visible=True, value=filename, label=sample_id) ] + [ child for child in children ]

      else:
        return [ gr.update(visible=False) ] * 11
    
    sample_id.change(update_sample_wavs,
      inputs = project_defining_inputs + [sample_id],
      outputs = [ sample_audio ] + sample_children_audios,
    )

    # On load, load general data from server
    def load_general_data(*args):
      print('Loading general data')
      # Convert general inputs to a dict so we can return it
      out = {}
      for i, value in enumerate(args):
        print(f'Loading {i} = {value}')
        out[UI.general_inputs[i]] = value
      print(f'General data: {out}')
      # Read 'settings.json' from root folder if it exists
      if os.path.exists('settings.json'):
        print('Found settings.json')
        with open('settings.json', encoding='utf-8') as f:
          # For every key, find the corresponding input and set its value
          for key, value in json.load(f).items():
            out[UI.__dict__[key]] = value
      else:
        print('No settings.json found')
      return out
    
    ui.load(load_general_data, inputs=general_inputs, outputs=general_inputs)



  def __init__(self):
    self.ui.launch()

  # Set default values directly
  base_folder.value = 'G:/Мой диск/AI music'  #@param {type:"string"}

  project_name.value = ''                     #@param {type:"string"}
  project_name.choices = get_project_names(base_folder.value)

  artist.value = 'Unknown'                    #@param {type:"string"}
  genre.value = 'Unknown'                     #@param {type:"string"}
  lyrics.value = ''                           #@param {type:"string"}
  duration.value = 200                        #@param {type:"number"}

  step.value = 'Start'                        #@param ['Start', 'Continue']

  sample_id.value = ''                        #@param {type:"string"}
  prime_id.value = ''                         #@param {type:"string"}

  generation_length.value = 3                 #@param {type:"number"}

if __name__ == '__main__':
  UI()