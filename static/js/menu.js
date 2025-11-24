document.addEventListener('DOMContentLoaded', function() {
        // Récupérer le chemin de l'URL actuelle (sans le domaine)
        const currentPath = window.location.pathname;
        
        // Retirer la classe 'active' de tous les liens
        document.querySelectorAll('.sidebar .nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        // Mapping entre les chemins URL et les classes CSS
        const urlToClassMap = {
            '/dashboard': 'dashboard',
            '/cleapi': 'cleapi',
            '/donneesentrainement': 'donneesentrainement',
            '/entrainementmodeles': 'entrainementmodeles',
            '/integrationfeedback': 'integrationfeedback',
            '/validation': 'validation',
            '/testeurcommande': 'testeurcommande'
        };
        
        // Trouver la classe correspondant à l'URL actuelle
        let activeClass = null;
        for (const [path, className] of Object.entries(urlToClassMap)) {
            if (currentPath.includes(path)) {
                activeClass = className;
                break;
            }
        }
        console.log(activeClass)
        
        // Ajouter la classe 'active' au lien correspondant
        if (activeClass) {
            const activeLink = document.querySelector(`.sidebar .nav-link.${activeClass}`);
            if (activeLink) {
                activeLink.classList.add('active');
            }
        }
    });