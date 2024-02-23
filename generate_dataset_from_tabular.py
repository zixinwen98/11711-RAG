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