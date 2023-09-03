from params import base_path

def get_sample_filename(project_name, sample_id, cut_out, last_n_sec, upsample_rendering, combine_levels, invert_center):

    filename = f'{base_path}/{project_name}/rendered/{sample_id} '

    # Add cutout/preview suffixes, replacing dots with underscores (to avoid confusion with file extensions)

    def replace_dots_with_underscores(number):
      return str(number).replace('.', '_')

    if cut_out:
      filename += f'cut {replace_dots_with_underscores(cut_out)} '
    if last_n_sec:
      filename += f'last {replace_dots_with_underscores(last_n_sec)} '

    # Add lowercase of upsample rendering option
    if upsample_rendering:
      filename += f'r{upsample_rendering} '

    if combine_levels:
      filename += f'combined '

    if invert_center:
      filename += 'inverted '

    return filename