const form = document.querySelector('form')

form.addEventListener('submit', (e) => {
    e.preventDefault();

    const formData = new FormData();

    const files = document.querySelector('[type=file]').files;
    formData.append('file', files[0]);

    const requestBody = {
        method: 'POST',
        body: formData,
    }

    fetch('song', requestBody).then((response) => {
        response.json().then((json) => {
            console.log(json)
        });
    });
})