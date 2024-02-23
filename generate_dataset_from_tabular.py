from utils import *

tabular_data = ['lti_staff.json', 'lti_faculty.json', 'university_calendar.json', 'course_schedule.json', 'faculty_publication.json']

def lti_staff(data, num_samples=50, verbose_p = 0.5):
    '''
    data: list of string formatted in json 
    num_samples: number of samples to generate
    verbose_p: probability of generating a verbose answer
    return: 
    a list of dictionary with key: question, answer, context
    '''
    contact = ['Email', 'Phone', 'Office']
    template = {
        "What is the [item] of LTI staff [name]?":'belonging',
        "Please tell me the [item] of LTI staff [name] based on the background information: ": 'belonging',
        "How can I contact LTI staff [name]?": 'contact',
        "Please provide the contact information of LTI staff [name]": 'contact',
        "Is [name] [title] of LTI?, Please answer yes or no: ": 'binary_qa'
    }
    result = []
    for _ in range(num_samples):
        data_dict = dict() 
        data_string = random.choice(data)
        name, title = ' '.join(data_string.split(' ')[:2]), data_string[2: data_string.index('Email')].strip()
        avaliable_info = [c for c in contact if c in data_string]

        question = random.choice(list(template.keys()))
        question_type = template[question]
        
        if question_type == 'belonging':
            item = random.choice(avaliable_info)

            #find answer
            idx = list(sorted([data_string.index(i) for i in avaliable_info])) 
            info = [data_string[idx[i]:idx[i+1]] for i in range(len(idx)-1)] + [data_string[idx[-1]:]]

            for i in info:
                if item in i:
                    answer = " ".join(i.split(' ')[1:])
                    break
            
            if random.random() < verbose_p:
                answer = f"{name}'s {item} is {answer}"

            context = data_string
            question = question.replace('[item]', item)
            question = question.replace('[name]', name)
        
        elif question_type == 'contact':
            name = name
            answer = data_string[data_string.index("Email"): ]
            context = data_string
            question = question.replace('[name]', name)
        
        elif question_type == 'binary_qa':
            if random.random() < 0.5:
                another_title = random.choice(data)
                another_title = another_title[2: another_title.index('Email')].strip()
                
                if title != another_title:
                    answer = 'No'
                else:
                    answer = 'Yes'
                title = another_title
                answer = 'No'
                
            else:
                answer = 'Yes'
                title  = title
            question = question.replace('[title]', title)
            question = question.replace('[name]', name)
            context = data_string
        
        data_dict['question'] = question
        data_dict['answer'] = answer.strip()
        data_dict['context'] = context
        result.append(data_dict)
    return result


def lti_faculty(data, num_samples=50, verbose_p = 0.5):
    '''
    data: list of string formatted in json 
    num_samples: number of samples to generate
    verbose_p: probability of generating a verbose answer
    return: 
    a list of dictionary with key: question, answer, context
    '''
    contact = ['Email', 'Phone', 'Office', 'Research Areas', 'Research Areas']
    template = {
        "What is the [item] of LTI faculty [name]?":'belonging',
        "Please tell me the [item] of LTI faculty [name] based on the background information: ": 'belonging',
        "How can I contact LTI faculty [name]?": 'contact',
        "Please provide the contact information of LTI faculty [name]": 'contact',
        "Is [name] [title] of LTI?, Please answer yes or no: ": 'binary_qa'
    }
    result = []
    for _ in range(num_samples):
        data_dict = dict() 
        data_string = random.choice(data)
        name, title = ' '.join(data_string.split(' ')[:2]), data_string[2: data_string.index('Email')].strip()
        avaliable_info = [c for c in contact if c in data_string]

        question = random.choice(list(template.keys()))
        question_type = template[question]
        
        if question_type == 'belonging':
            item = random.choice(avaliable_info)

            #find answer
            idx = list(sorted([data_string.index(i) for i in avaliable_info])) 
            info = [data_string[idx[i]:idx[i+1]] for i in range(len(idx)-1)] + [data_string[idx[-1]:]]
       
            for i in info:
                if item in i:
                    answer = " ".join(i.split(' ')[1:])
                    if 'Areas' in answer:
                        answer = answer[answer.index('Areas') + 7:]
                    break
            
            if random.random() < verbose_p:
                answer = f"{name}'s {item} is {answer}"

            context = data_string
            question = question.replace('[item]', item)
            question = question.replace('[name]', name)
        
        elif question_type == 'contact':
            name = name
            try:
                answer = data_string[data_string.index("Email"): data_string.index("Research Areas")]
            except:
                answer = data_string[data_string.index("Email"):]
            context = data_string
            question = question.replace('[name]', name)
        
        elif question_type == 'binary_qa':
            if random.random() < 0.5:
                another_title = random.choice(data)
                another_title = another_title[2: another_title.index('Email')].strip()
                
                if title != another_title:
                    answer = 'No'
                else:
                    answer = 'Yes'
                title = another_title
                answer = 'No'
                
            else:
                answer = 'Yes'
                title  = title
            question = question.replace('[title]', title)
            question = question.replace('[name]', name)
            context = data_string
        
        data_dict['question'] = question
        data_dict['answer'] = answer.strip()
        data_dict['context'] = context
        result.append(data_dict)
    return result

def university_calendar(data, num_samples=50, verbose_p = 0.5):
    
    template = {
        "According to background information, What happens in [date] on university calendar?":'date-to-event',
        "Based on background information, What is the specific event listed for [date], in [semester]":'date-to-event',
        "According to background information, When is the [event] in [semester]?":"event-to-date",
        "Based on the given information, Does [semester] [event] happen at [date]":"binary_qa"
    }
    result = []
    for _ in range(num_samples):
        data_dict = dict() 
        data_string = random.choice(data)
        semester = data_string[:data_string.index(":")]
        event = data_string.split("Event: ")[1]
        date = data_string.split("Date: ")[1].split(" ")[0]

        question = random.choice(list(template.keys()))
        question_type = template[question]
        
        if question_type == 'date-to-event':
            #find answer
            answer = event
            context = data_string
            question = question.replace('[date]', date)
            if '[semester]' in question:
                question = question.replace('[semester]', semester)
            if random.random() < verbose_p:
                answer = f"The event happening on {date} is {event}"
        
        elif question_type == 'event-to-date':
            answer = date
            context = data_string
            question = question.replace('event', event)
            if '[semester]' in question:
                question = question.replace('[semester]', semester)
            if random.random() < verbose_p:
                answer = f"The event {event} happens on {date}"
        
        elif question_type == 'binary_qa':
            if random.random() < 0.5:
                another_event = random.choice(data)
                if another_event == data_string: 
                    question = question.replace('[date]', date)
                    answer = 'Yes'
                else:
                    another_date = another_event.split("Date: ")[1].split(" ")[0]
                    question = question.replace('[date]', another_date)
                    answer = 'No'
            else:
                question = question.replace('[date]', date)
                answer = 'Yes'
            question = question.replace('[event]', event)
            question = question.replace('[semester]', semester)
            context = data_string
        
        data_dict['question'] = question
        data_dict['answer'] = answer.strip()
        data_dict['context'] = context
        result.append(data_dict)
    return result

def course_schedule(data, num_samples=50, verbose_p = 0.5):
    '''
    Spring offering: Course: 48105 
    Title: Architecture Design Studio: Poiesis Studio 2 Units: 15.0 
    Lec/Sec: Lec 
    Days: MWF 
    Begin: 02:00PM End: 04:50PM 
    Bldg/Room: MM A14 
    Location: Pittsburgh, Pennsylvania 
    Instructor(s): Yang
    '''
    template = {
        "What is the units of course [title] offered in [semester]?":"units",
        "Who is the instructor(s) of course [title] [lec/sec] offered in [semester]?":"instructor",
        "On which days does the class [title] [lec/sec] offered in [semester] meet?":'days',
        "At what time does the class [title] [lec/sec] offered in [semester] begin?" : "time",
        "When does the class [title] [lec/sec] offered in [semester] end?":"time",
        "Which campus is class [title] [lec/sec] offered in Pittsburgh in [semester]?":"location",
        "Where does the class [title] [lec/sec] offered in [semester] meet?":"room",
    }
    result = []
    for _ in range(num_samples):
        data_dict = dict()
        data_string = random.choice(data)
        title = data_string.split("Title: ")[1].split("Units")[0].strip()
        semester = data_string[:2]
        time_begin = data_string.split("Begin: ")[1].split("End")[0].strip()
        time_end = data_string.split("End: ")[1].split("Bldg/Room")[0].strip()
        location = data_string.split("Location: ")[1].split("Instructor(s)")[0].strip()
        instructor = data_string.split("Instructor(s): ")[1].strip()
        days = data_string.split("Days: ")[1].split("Begin")[0]
        units = data_string.split("Units: ")[1].split("Lec/Sec")[0]
        lec_sec = data_string.split("Lec/Sec: ")[1].split("Days")[0]
        room = data_string.split("Bldg/Room: ")[1].split("Location")[0].strip()

        question = random.choice(list(template.keys()))
        question_type = template[question]

        if 'Section' in lec_sec:
                lec_sec = 'Recitation ' + lec_sec

        #skip the loop if any of the information is equal to 'TBA'
        if 'TBA' in data_string or 'To be announced' in data_string:
            continue

        if question_type == 'units':
            answer = units 
            context = data_string

            if random.random() < verbose_p:
                answer = f"The units of course {title} is {units}"
                question = "Based on the background information, " + question
        
        if question_type in ['days', 'instructor', 'room']:
            if question_type == 'days':
                answer = days
            elif question_type == 'instructor':
                answer = instructor
            else:
                answer = room
            context = data_string
            question = question.replace('[lec/sec]', lec_sec)
        
        if question_type == 'time':
            if 'begin' in question:
                answer = time_begin
            else:
                answer = time_end
            context = data_string
            question = question.replace('[lec/sec]', lec_sec)
        
        if question_type == 'location': 
            answer = location
            context = data_string
            question = question.replace('[lec/sec]', lec_sec)

        question = question.replace('[title]', title)
        question = question.replace('[semester]', semester)
        data_dict['question'] = question    
        data_dict['answer'] = answer.strip()
        data_dict['context'] = context
        result.append(data_dict)

    return result

def faculty_publication(data, num_samples=50, verbose_p=0.5):
    '''
    Author: Yonatan Bisk 
    Title: HomeRobot: An Open Source Software Stack for Mobile Manipulation Research 
    Publication year: 2024 
    '''

    template = {
        "Which LTI faculty published the paper [title] in [year]?":"author",
        "Who is the author of the LTI paper [title] published in [year]?":"author",
        "What is the title of the paper published by LTI faculty [author] in [year]?":"title",
        "What is the abstract of the paper [title] published by LTI faculty [author] in [year]?":"abstract",
        "Who are the coauthors of the paper [title] published by LTI faculty [author] in [year]?":"coauthor",
    }

    result = []

    for _ in range(num_samples):
        data_dict = dict()
        data_string = random.choice(data)
        author = data_string.split("Author: ")[1].split("Title")[0].strip()
        title = data_string.split("Title: ")[1].split("Publication")[0].strip()
        year = data_string.split("Publication year: ")[1].split("Coauthors")[0].strip()
        coauthors = data_string.split("Coauthors: ")[1].split("Abstract")[0].strip()
        try:
            abstract = data_string.split("Abstract: ")[1].strip()
        except:
            abstract = None

        question = random.choice(list(template.keys()))
        question_type = template[question]

        if question_type == 'author':
            answer = author
            question = question.replace('[title]', title)
            context = data_string
   
        if question_type == 'title':
            answer = title
            question = question.replace('[author]', author)
            context = data_string
        
        if question_type == 'abstract':
            answer = abstract
            if abstract is None:
                continue 
            question = question.replace('[title]', title)
            question = question.replace('[author]', author)
            context = data_string
        
        if question_type == 'coauthor':
            answer = coauthors
            question = question.replace('[title]', title)
            question = question.replace('[author]', author)
            context = data_string
        
        question = question.replace('[year]', year)
        data_dict['question'] = question
        data_dict['answer'] = answer.strip()
        data_dict['context'] = context
        result.append(data_dict)

    return result

def generate_dataset_from_tabular(num_samples_each=50, verbose_p=0.5):
    final = []
    for fname in tabular_data:
        path = f'data/cmu/{fname}'
        data = json.load(open(path))
        if 'lti_staff' in fname:
            result = lti_staff(data, num_samples=num_samples_each, verbose_p=verbose_p)
        elif 'lti_faculty' in fname:
            result = lti_faculty(data, num_samples=num_samples_each, verbose_p=verbose_p)
        elif 'university_calendar' in fname:
            result = university_calendar(data, num_samples=num_samples_each, verbose_p=verbose_p)
        elif 'course_schedule' in fname:
            result = course_schedule(data, num_samples=num_samples_each, verbose_p=verbose_p)
        elif 'faculty_publication' in fname:
            result = faculty_publication(data, num_samples=num_samples_each, verbose_p=verbose_p)
        else:
            raise ValueError("Invalid file name")
        final.extend(result)
    return final

if __name__ == '__main__':
    seed_everything()
    result = generate_dataset_from_tabular()
    with open('data/automated_questions.json', 'w') as f:
        json.dump(result, f)
    print("Done!")