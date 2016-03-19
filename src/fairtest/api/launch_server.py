import os
from flask import Flask, request, render_template, redirect, url_for
from flask import send_from_directory
from werkzeug import secure_filename

UPLOAD_FOLDER = '/tmp/fairtest'
ALLOWED_EXTENSIONS = set(['csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def make_tree(path):
    tree = dict(name=os.path.basename(path), children=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=name))
    return tree


@app.route('/fairtest/demo_app', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
          file = request.files['file']
        except Exception, error:
          print error
          raise
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template("upload.html", tree=make_tree(app.config['UPLOAD_FOLDER']))
    return render_template("upload.html", tree=make_tree(app.config['UPLOAD_FOLDER']))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print filename
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
  app.run(debug=True)


